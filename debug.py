# from pyserini.search import get_topics, get_qrels
#
# topics = get_topics('msmarco-passage-dev-subset')
# qrels = get_qrels('msmarco-passage-dev-subset')
#
# # Inspect the first few keys and types from each
# print("=== TOPICS (first 3) ===")
# for i, (k, v) in enumerate(topics.items()):
#     print(f"  key={k!r}  type={type(k).__name__}  value={v}")
#     if i >= 2:
#         break
#
# print("\n=== QRELS (first 3) ===")
# for i, (k, v) in enumerate(qrels.items()):
#     print(f"  key={k!r}  type={type(k).__name__}  value={v}")
#     if i >= 2:
#         break
#
# # Try to match the first topic qid against qrels using both int and str
# first_qid = next(iter(topics))
# print(f"\n=== LOOKUP TEST for qid={first_qid!r} ===")
# print(f"  qrels.get(first_qid):        {qrels.get(first_qid)}")
# print(f"  qrels.get(str(first_qid)):   {qrels.get(str(first_qid))}")
# print(f"  qrels.get(int(first_qid)):   {qrels.get(int(first_qid)) if str(first_qid).isdigit() else 'N/A'}")

from pyserini.search import get_topics
topics = get_topics('msmarco-passage-dev-subset')
for qid in [1102330, 160885, 1091115, 359349, 490595]:
    print(qid, qid in topics, topics.get(qid, {}).get('title', 'NOT FOUND'))
