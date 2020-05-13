query = "Fauci told CNN earlier Sunday that the US could see millions of cases." # pass the query

query_embedding = embed([query])

nns = ann.get_nns_by_vector(query_embedding[0], 10)

results = []
for n in nns:
  result = json.loads(r.get(str(n)))
  results.append(result)
