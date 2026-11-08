"""Kiểm thử giao diện web (hỏi tiếp, cuộc hỏi mới, cảnh báo trích dẫn) bằng trình duyệt thật + API giả.

    python -m tests.test_web        (cần playwright + chromium; không chạy trong Jenkins)

API /ask được giả bằng page.route: ghi lại body mỗi lần gửi để kiểm tra history.
"""
import json
import pathlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from playwright.sync_api import sync_playwright

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"
API = "https://legal-api.oci.lexsy.pro.vn/ask"     # lấy từ <meta name="legal-api"> trong index.html

SRC = [{"n": 1, "citation": "Điều 459 Bộ luật 91/2015/QH13", "title": "Điều 459. Tặng cho bất động sản",
        "key": "91/2015/qh13|459", "effective_date": "2017-01-01", "text": "Điều 459. Tặng cho bất động sản\n1. Nội dung."},
       {"n": 2, "citation": "Điều 462 Bộ luật 91/2015/QH13", "title": "Điều 462. Tặng cho tài sản có điều kiện",
        "key": "91/2015/qh13|462", "effective_date": "2017-01-01", "text": "Điều 462. Tặng cho tài sản có điều kiện\n1. Nội dung."}]


def reply(n, req):
    body = json.loads(req.post_data)
    follow = bool(body.get("history"))
    data = {"ok": True, "answer": f"Trả lời lượt {n} [1].\n\nThông tin tham khảo, không thay tư vấn pháp lý.",
            "sources": SRC, "used_sources": [1], "trace_id": "a" * 32,
            "standalone_question": ("Bố mẹ đòi lại đất đã tặng cho khi con không phụng dưỡng" if follow else body["question"]),
            "rerank_query": "Đòi lại quyền sử dụng đất đã tặng cho con" if n == 1 else "",
            "unverified_citations": ["Điều 500"] if n == 2 else []}
    return data


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(WEB)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/index.html"
    sent = []

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page()
        page.route("https://fonts.googleapis.com/**", lambda r: r.abort())
        page.route("https://fonts.gstatic.com/**", lambda r: r.abort())
        page.route("**/feedback", lambda r: r.fulfill(status=204))

        def handle(route):
            sent.append(json.loads(route.request.post_data))
            route.fulfill(status=200, content_type="application/json", body=json.dumps(reply(len(sent), route.request)))
        page.route(API, handle)
        page.goto(url)

        turns = page.locator(".turn")
        # lượt 1: câu hỏi đầu, history rỗng
        page.fill("#q", "Bố mẹ đã sang tên đất cho tôi, nay đòi lại để chia cho em tôi được không?")
        page.press("#q", "Enter")
        turns.nth(0).locator(".answer").wait_for()
        assert sent[0]["history"] == [], sent[0]
        und0 = turns.nth(0).locator(".understood")                      # câu đầu: hiện tiêu đề vấn đề pháp lý
        assert und0.is_visible() and "Đòi lại quyền sử dụng đất đã tặng cho con" in und0.inner_text(), und0.inner_text()
        assert turns.nth(0).locator(".unverified").is_hidden()
        assert page.locator("#new-chat").is_visible()

        # lượt 2: hỏi tiếp -> history có lượt 1, đã bỏ [n] và câu nhắc cuối
        page.fill("#q", "Thế nếu tôi không phụng dưỡng bố mẹ thì sao?")
        page.press("#q", "Enter")
        turns.nth(1).locator(".answer").wait_for()
        h = sent[1]["history"]
        assert len(h) == 1 and h[0]["q"].startswith("Bố mẹ đã sang tên") and h[0]["a"] == "Trả lời lượt 1.", h
        und = turns.nth(1).locator(".understood")
        assert und.is_visible() and "không phụng dưỡng" in und.inner_text()
        unv = turns.nth(1).locator(".unverified")
        assert unv.is_visible() and "Điều 500" in unv.inner_text()

        # lượt 3..6: chỉ gửi tối đa 4 lượt gần nhất
        for i in range(3, 7):
            page.fill("#q", f"Câu hỏi tiếp số {i} về chuyện đất đai")
            page.press("#q", "Enter")
            turns.nth(i - 1).locator(".answer").wait_for()
        assert len(sent[5]["history"]) == 4 and sent[5]["history"][0]["a"].startswith("Trả lời lượt 2"), sent[5]["history"]

        # 👎 gửi kèm câu hỏi đầy đủ
        with page.expect_request("**/feedback") as fb:
            turns.nth(1).locator('.vote-btn[data-vote="down"]').click()
        f = json.loads(fb.value.post_data)
        assert f["vote"] == "down" and f["standalone"].startswith("Bố mẹ đòi lại đất") and f["question"].startswith("Thế nếu"), f

        # cuộc hỏi mới: xoá hết, câu sau history rỗng
        page.click("#new-chat")
        assert turns.count() == 0 and page.locator("#intro").is_visible() and page.locator("#new-chat").is_hidden()
        page.fill("#q", "Công ty nợ lương 3 tháng không trả, tôi phải làm gì?")
        page.press("#q", "Enter")
        turns.nth(0).locator(".answer").wait_for()
        assert sent[-1]["history"] == [], sent[-1]

        # gửi lại một lượt lỗi: dùng đúng ngữ cảnh lúc hỏi, lượt lỗi không vào history
        page.unroute(API)
        page.route(API, lambda r: r.fulfill(status=502, content_type="application/json", body='{"detail": "Dịch vụ trả lời lỗi 500"}'))
        page.fill("#q", "Hỏi tiếp nhưng n8n đang lỗi thì sao?")
        page.press("#q", "Enter")
        turns.nth(1).locator(".failure").wait_for(state="visible")
        page.unroute(API)
        page.route(API, handle)
        turns.nth(1).locator(".retry").click()
        turns.nth(1).locator(".answer").wait_for()
        assert len(sent[-1]["history"]) == 1 and sent[-1]["history"][0]["q"].startswith("Công ty nợ lương"), sent[-1]
        b.close()
    srv.shutdown()
    print(f"✓ giao diện: hỏi tiếp gửi history (tối đa 4 lượt, bỏ [n]), hiện câu hiểu được + cảnh báo trích dẫn, "
          f"👎 gửi câu đầy đủ, cuộc hỏi mới xoá ngữ cảnh, gửi lại dùng đúng ngữ cảnh ({len(sent)} request)")


if __name__ == "__main__":
    main()
