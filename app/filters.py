def build_where_clause(
    arc: str | None, session: str | None, tag: str | None
) -> dict | None:
    clauses: list[dict] = []
    if arc and arc != "All":
        clauses.append({"arc": arc})
    if session and session != "All":
        clauses.append({"session": str(session)})
    if tag and tag != "All":
        clauses.append({"tags": {"$contains": tag}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
