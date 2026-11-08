// Kiểm thử các biểu thức trong workflow n8n (n8n/legal-ask.nodes.json) bằng Node, không cần n8n.
//   node tests/test_n8n.js
const fs = require('fs'), path = require('path');
const N = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'n8n', 'legal-ask.nodes.json'))).nodes;
const node = n => ({ parameters: N[n], value: N[n].jsonBody });
const ex = (s) => s.slice(3).replace(/\}\}\s*$/, '');
const run = (expr, $json, nodes) => Function('$json', '$', `return (${ex(expr)});`)($json, n => ({ item: { json: nodes[n] } }));
const asg = (n, name) => node(n).parameters.assignments.assignments.find(a => a.name === name).value;
const assert = (c, m) => { if (!c) { console.error('FAIL', m); process.exit(1); } };

// 1. Chuẩn hoá: history lọc/giới hạn
const body = { body: { question: '  Thế nếu tôi không phụng dưỡng? ', history: [1,2,3,4,5].map(i => ({ q: 'q' + i, a: 'a'.repeat(2000) })).concat([{ q: '' }]) } };
const H = run(asg('Chuẩn hoá câu hỏi', 'history'), body, {});
assert(H.length === 3 || H.length === 4, 'len ' + H.length);       // slice(-4) trước rồi lọc q rỗng -> 3
assert(H.every(t => t.a.length === 1500), 'cắt a');
assert(run(asg('Chuẩn hoá câu hỏi', 'history'), { body: { question: 'x' } }, {}).length === 0, 'không có history');
assert(run(asg('Chuẩn hoá câu hỏi', 'history'), { body: { question: 'x', history: 'rác' } }, {}).length === 0, 'history sai kiểu');
const HT = run(asg('Chuẩn hoá câu hỏi', 'history_text'), { body: { question: 'x', history: [{ q: 'Bố mẹ sang tên đất', a: 'Không đòi được' }] } }, {});
assert(HT === 'Lượt 1\nNgười hỏi: Bố mẹ sang tên đất\nTrợ lý: Không đòi được', HT);

// 2. Câu hỏi ngắn: tách 2 dòng
const first = { question: 'Bố mẹ đã sang tên đất cho tôi, nay đòi lại được không?', k: 5, history: [] };
const fol = { question: 'Thế nếu tôi không phụng dưỡng thì sao?', k: 5, history: [{ q: 'Bố mẹ đã sang tên đất cho tôi, nay đòi lại được không?', a: '...' }] };
const out1 = 'Câu hỏi đầy đủ: -\nTiêu đề: Đòi lại quyền sử dụng đất đã tặng cho con';
const out2 = 'Câu hỏi đầy đủ: "Bố mẹ tặng cho con đất, con không phụng dưỡng thì bố mẹ có đòi lại được không?"\nTiêu đề: Đòi lại đất tặng cho khi con không phụng dưỡng.';
const Q = (text, n) => ({ q: run(asg('Câu hỏi ngắn', 'question'), { text }, { 'Chuẩn hoá câu hỏi': n }), rq: run(asg('Câu hỏi ngắn', 'rerank_query'), { text }, { 'Chuẩn hoá câu hỏi': n }) });
let r = Q(out1, first);
assert(r.q === first.question && r.rq === 'Đòi lại quyền sử dụng đất đã tặng cho con', JSON.stringify(r));
r = Q(out2, fol);
assert(r.q === 'Bố mẹ tặng cho con đất, con không phụng dưỡng thì bố mẹ có đòi lại được không?' && r.rq === 'Đòi lại đất tặng cho khi con không phụng dưỡng', JSON.stringify(r));
r = Q('', fol);                                   // LLM lỗi hẳn -> ghép câu trước + câu mới, rerank để API tự lo
assert(r.q.startsWith('Bố mẹ đã sang tên') && r.q.endsWith('thì sao?') && r.rq === '', JSON.stringify(r));
r = Q('Câu hỏi đầy đủ: -\nTiêu đề: Xử lý khi công ty nợ lương nhiều tháng', fol);   // LLM coi là chuyện khác -> dùng nguyên câu mới
assert(r.q === fol.question && r.rq === 'Xử lý khi công ty nợ lương nhiều tháng', JSON.stringify(r));
r = Q('Tiêu đề: Đòi lại quyền sử dụng đất đã tặng cho con', first);   // model cũ chỉ trả 1 dòng
assert(r.rq === 'Đòi lại quyền sử dụng đất đã tặng cho con', JSON.stringify(r));
r = Q('Đòi lại quyền sử dụng đất đã tặng cho con', first);             // không có nhãn
assert(r.rq === 'Đòi lại quyền sử dụng đất đã tặng cho con', JSON.stringify(r));
r = Q('Câu hỏi đầy đủ: -', first);                                     // thiếu dòng tiêu đề
assert(r.rq === '', JSON.stringify(r));

// 3. body gọi API: câu đầy đủ + câu ngắn tách riêng
const b = JSON.parse(run(node('Tìm Điều luật (API /search)').value, { question: 'Q đầy đủ', rerank_query: 'tiêu đề', k: 5 }, {}));
assert(b.question === 'tiêu đề. Q đầy đủ' && b.rerank_query === 'tiêu đề' && b.max_chars === 2500, JSON.stringify(b));
const b2 = JSON.parse(run(node('Tìm Điều luật (API /search)').value, { question: 'Q đầy đủ', rerank_query: '', k: 5 }, {}));
assert(b2.question === 'Q đầy đủ' && b2.rerank_query === '', JSON.stringify(b2));

// 4. trích dẫn ngoài nguồn
const src = { body: { sources: [{ n: 1, citation: 'Điều 459 Bộ luật 91/2015/QH13', title: 'Điều 459. Tặng cho bất động sản' }, { n: 2, citation: 'Điều 462 Bộ luật 91/2015/QH13', title: 'Điều 462' }] } };
const nodes = { 'Tìm Điều luật (API /search)': src, 'Câu hỏi ngắn': { question: 'q', rerank_query: 'x' } };
const unv = t => run(asg('Định dạng câu trả lời', 'unverified_citations'), { text: t }, nodes);
assert(JSON.stringify(unv('Theo Điều 459 [1] và Điều 462 [2].')) === '[]', 'đủ nguồn');
assert(JSON.stringify(unv('Căn cứ các điều 457, 459 và 462; các điều 117, 122–131 về hiệu lực. Khoản 1 Điều 500.')) === JSON.stringify(['Điều 457', 'Điều 117', 'Điều 122', 'Điều 131', 'Điều 500']), JSON.stringify(unv('Căn cứ các điều 457, 459 và 462; các điều 117, 122–131 về hiệu lực. Khoản 1 Điều 500.')));
const reply = run(asg('Định dạng câu trả lời', 'reply_text'), { text: 'Theo Điều 457 [1].' }, nodes);
assert(reply.includes('⚠ Câu trả lời có nhắc Điều 457') && reply.includes('[1] Điều 459'), reply);
const used = run(asg('Định dạng câu trả lời', 'used_sources'), { text: 'a [2] b [1] c [9]' }, nodes);
assert(JSON.stringify(used) === '[1,2]', used);

// 5. prompt trả lời: có lịch sử + câu hiểu đầy đủ
const at = Function('$json', '$', '$now', `return (${ex(N['Trả lời theo nguồn'].text)});`)(
  src, n => ({ item: { json: { 'Câu hỏi ngắn': { asked: 'Thế nếu...?', question: 'Câu đầy đủ', is_follow_up: true }, 'Chuẩn hoá câu hỏi': { history_text: 'Lượt 1\nNgười hỏi: A' } }[n] } }), { toISODate: () => '2026-11-08', toFormat: () => '08/11/2026' });
assert(at.startsWith('Bạn là trợ lý tra cứu pháp luật Việt Nam. Hôm nay là 08/11/2026.') && at.includes('=====\n\nLỊCH SỬ HỘI THOẠI') && at.indexOf('Quy tắc') < at.indexOf('CÁC ĐIỀU LUẬT TÌM ĐƯỢC CHO') && at.includes('[1] Điều 459') && at.includes('CÂU HỎI MỚI (câu hỏi tiếp):\nThế nếu...?\n(Hiểu đầy đủ là: Câu đầy đủ)'), at);
// 6. hướng dẫn viết lại nằm trong tin nhắn người dùng (proxy bỏ system message)
const rw = Function('$json', `return (${ex(N['Viết lại câu hỏi'].text)});`)({ question: 'Thế còn con thì sao?', history_text: 'Lượt 1\nNgười hỏi: A' });
assert(rw.startsWith('Bạn giúp hệ thống tra cứu') && rw.endsWith('LỊCH SỬ HỘI THOẠI:\nLượt 1\nNgười hỏi: A\n\nCÂU HỎI MỚI:\nThế còn con thì sao?'), rw.slice(-120));
assert(N['Viết lại câu hỏi'].messages.messageValues.length === 0 && N['Trả lời theo nguồn'].messages.messageValues.length === 0, 'không dùng system message');
console.log('✓ biểu thức n8n: chuẩn hoá history, tách câu đầy đủ/tiêu đề (+ dự phòng), body API, phát hiện trích dẫn ngoài nguồn, prompt trả lời');
