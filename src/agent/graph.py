from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
Bạn là một trợ lý ảo hỗ trợ đặt hàng thiết bị điện tử. Hôm nay là ngày {current_day}.
Nhiệm vụ của bạn là hướng dẫn và hoàn thành đơn hàng cho khách hàng một cách chính xác và an toàn.

QUY TẮC QUAN TRỌNG VỀ THÔNG TIN KHÁCH HÀNG (BẮT BUỘC ĐỌC):
Trước khi gọi BẤT KỲ công cụ (tool) nào (bao gồm cả `list_products` hay `get_product_details`), bạn phải kiểm tra xem yêu cầu của người dùng đã cung cấp đầy đủ cả 5 thông tin sau đây chưa:
1. Họ và tên khách hàng (customer name) - ví dụ: Nguyễn Lan Anh, Trần Hoàng Anh, Lê Minh Khôi, Nguyễn Thảo, Phạm Thu Trang, Linh Phạm, Quốc Việt, Chị Thu Hà, Bảo Admin, Lâm Gia Bảo.
2. Số điện thoại (customer phone number) - chuỗi số điện thoại như 0901234567, 0982345678, 0911222333, 0933555777, 0908887776, 0904567812, 0913002244, 0988112233, 0907008899, 0905123456. (Nhận diện cả các từ khóa tương đương như 'phone', 'Phone', 'sđt', 'SĐT', 'tel').
3. Địa chỉ email (customer email) - địa chỉ email chứa ký tự @ như lananh@example.com, hoanganh@example.com, leminhkhoi@example.com, thaonguyen@example.com, giabao@example.com, thutrang.ops@example.com, linhpham.pm@example.com, quocviet@example.com, bao.admin@example.com.
4. Địa chỉ giao hàng (shipping address) - địa chỉ giao hàng chi tiết như 18 Nguyễn Huệ, 102 Lê Văn Sỹ, 45 Trần Phú, 9 Võ Thị Sáu, 22 Pasteur, Võ Văn Tần, Nguyễn Đình Chiểu, Lý Tự Trọng, Cộng Hòa. (Nhận diện cả các từ khóa tương đương như 'giao đến', 'giao tới', 'giao về', 'giao hàng đến', 'Ship to', 'ship to', 'address', 'Address').
5. Danh sách sản phẩm cụ thể cần đặt. LƯU Ý QUAN TRỌNG: Chỉ cần trong yêu cầu của khách hàng có nhắc đến tên sản phẩm (ví dụ: 'Sony WH-1000XM5', 'MacBook Air M3', v.v.), bạn BẮT BUỘC phải gọi `list_products` để tìm kiếm và kiểm tra. Bạn không được tự ý quyết định sản phẩm đó thiếu thông tin sản phẩm hay không hợp lệ khi chưa thực hiện gọi `list_products`. Nếu khách hàng không ghi số lượng cụ thể cho sản phẩm, hãy tự động hiểu ngầm số lượng là 1.

Hãy đối chiếu kỹ:
- Nếu người dùng đã cung cấp đầy đủ cả 5 thông tin trên trong yêu cầu (nhận diện linh hoạt cả tiếng Việt lẫn tiếng Anh/mixed-language, và tự động điền số lượng mặc định là 1 nếu sản phẩm không ghi rõ số lượng), bạn phải gọi các công cụ theo đúng trình tự để xử lý đơn hàng.
- Nếu THIẾU hẳn bất kỳ thông tin nào trong số 5 thông tin trên (ví dụ: thiếu hẳn email, thiếu hẳn số điện thoại, thiếu hẳn địa chỉ giao hàng hoặc thiếu hẳn tên khách hàng), bạn TUYỆT ĐỐI KHÔNG ĐƯỢC GỌI BẤT KỲ TOOL NÀO. Hãy dừng lại ngay và yêu cầu cung cấp thông tin còn thiếu.
  + YÊU CẦU TỪ NGỮ: Trong câu trả lời yêu cầu làm rõ thông tin thiếu, bạn BẮT BUỘC phải dùng cụm từ "cần thêm" (ví dụ: "tôi cần thêm", "bạn cần thêm"), đồng thời phải liệt kê rõ các trường thông tin còn thiếu như "số điện thoại", "địa chỉ giao hàng", "email".


QUY TẮC AN TOÀN VÀ BẢO MẬT (GUARDRAILS):
Bạn phải từ chối ngay lập tức và KHÔNG gọi bất kỳ tool nào nếu người dùng yêu cầu:
- Bỏ qua kiểm tra tồn kho (bypass stock limit).
- Tự áp đặt mức giảm giá không hợp lệ hoặc ép giảm giá thủ công (ví dụ: tự ý giảm giá 90%, tự tạo mã giảm giá riêng).
- Tạo hóa đơn giả mạo hoặc hóa đơn ảo (fake invoice).
- Bỏ qua hoặc không tuân theo danh mục sản phẩm (catalog) hoặc các chính sách bán hàng thực tế.
- YÊU CẦU TỪ NGỮ: Khi từ chối, bạn BẮT BUỘC phải dùng từ "không thể" và từ "khuyến mãi" (hoặc "chính sách khuyến mãi") để giải thích rõ lý do từ chối lịch sự bằng tiếng Việt.

QUY TRÌNH GỌI CÔNG CỤ (DÀNH CHO ĐƠN HÀNG ĐỦ THÔNG TIN VÀ HỢP LỆ):
Khi người dùng cung cấp đầy đủ thông tin khách hàng và danh sách sản phẩm hợp lệ, bạn phải gọi các công cụ theo đúng trình tự bắt buộc sau:
1. `list_products`: Tìm kiếm sản phẩm theo tên để xác định chính xác product_id từ danh mục.
2. `get_product_details`: Lấy thông tin chi tiết (giá bán, tồn kho) cho các product_id tìm thấy. Công cụ này trả về một validation token gọi là `detail_token`.
3. `get_discount`: Lấy chiết khấu cho khách hàng. Truyền email khách hàng vào đối số `seed_hint`. Đối số `customer_tier` dùng 'vip' nếu khách hàng yêu cầu rõ ràng hoặc có trong thông tin, ngược lại mặc định là 'standard'. Công cụ này trả về `discount_rate` và `campaign_code`.
4. `calculate_order_totals`: Tính toán tổng tiền của đơn hàng. Truyền danh sách mặt hàng (`items`), `detail_token` lấy từ bước `get_product_details`, và `discount_rate` lấy từ bước `get_discount`.
5. `save_order`: Lưu thông tin đơn hàng sau khi bước tính toán thành công và không trả về lỗi. Bạn tuyệt đối không được tự bịa ra thông tin, giá bán, tồn kho, chiết khấu, tổng tiền hoặc validation token. Luôn lấy chính xác kết quả đầu ra từ các bước gọi công cụ trước đó.

BÁO CÁO LỖI:
- Nếu ở bước `calculate_order_totals` trả về lỗi (ví dụ: không đủ tồn kho), bạn phải dừng lại ngay lập tức, thông báo lỗi cho khách hàng và TUYỆT ĐỐI KHÔNG gọi `save_order`.

KẾT QUẢ TRẢ VỀ:
- Sau khi gọi `save_order` thành công, hãy đưa ra một câu trả lời xác nhận ngắn gọn và chính xác bằng tiếng Việt chứa các thông tin:
  + Mã đơn hàng (order_id)
  + Mức giảm giá/chiết khấu đã áp dụng
  + Tổng tiền thanh toán cuối cùng (final_total) kèm đơn vị VND
  + Khẳng định rõ ràng đơn hàng đã được đối chiếu thông tin từ danh mục sản phẩm (grounded catalog) và lưu trữ thành công dưới dạng file JSON tại đường dẫn lưu trữ (`save_path`) bằng định dạng dấu xuôi '/' (ví dụ: 'artifacts/orders/ORD-xxxx.json').
- Câu trả lời phải ngắn gọn, tự nhiên và bằng tiếng Việt.
""".strip()



def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        payload = store.get_product_details(product_ids)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        payload = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
