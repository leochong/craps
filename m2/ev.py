from .montecarlo import FAIR

PLACE_ODDS = {
    2: 11 / 2,
    3: 11 / 4,
    4: 9 / 5,
    5: 7 / 5,
    6: 7 / 6,
    8: 7 / 6,
    9: 7 / 5,
    10: 9 / 5,
    11: 11 / 4,
    12: 11 / 2,
}

BUY_ODDS = {4: 2 / 1, 5: 3 / 2, 6: 6 / 5, 8: 6 / 5, 9: 3 / 2, 10: 2 / 1}
BUY_COMMISSION = 0.05


def place_ev(P, target, odds):
    p_n = P.get(target, 0.0)
    p_7 = P.get(7, 0.0)
    denom = p_n + p_7
    if denom <= 0 or odds <= 0:
        return None
    p_win = p_n / denom
    p_lose = p_7 / denom
    ev = p_win * odds - p_lose
    kelly = (p_win * (odds + 1.0) - 1.0) / odds
    return {
        "target": target,
        "odds": odds,
        "p_win": p_win,
        "p_lose": p_lose,
        "ev": ev,
        "kelly": max(0.0, kelly),
    }


def buy_ev(P, target, odds, commission=BUY_COMMISSION):
    p_n = P.get(target, 0.0)
    p_7 = P.get(7, 0.0)
    denom = p_n + p_7
    if denom <= 0 or odds <= 0:
        return None
    p_win = p_n / denom
    p_lose = p_7 / denom
    ev = p_win * odds - p_lose - commission
    kelly = (p_win * (odds + 1.0) - 1.0 - commission) / odds
    return {
        "target": target,
        "odds": odds,
        "p_win": p_win,
        "p_lose": p_lose,
        "ev": ev,
        "kelly": max(0.0, kelly),
    }


def ev_table(P, threshold=0.02, include_buy=False):
    rows = []
    for target, odds in sorted(PLACE_ODDS.items()):
        row = place_ev(P, target, odds)
        if row is not None:
            row["signal"] = row["ev"] > threshold
            rows.append(row)
    if include_buy:
        for target, odds in sorted(BUY_ODDS.items()):
            row = buy_ev(P, target, odds)
            if row is not None:
                row["signal"] = row["ev"] > threshold
                row["type"] = "buy"
                rows.append(row)
    return rows


def house_edge_reference(threshold=0.02):
    return ev_table(FAIR, threshold=threshold)


def best_opportunity(P, threshold=0.02):
    rows = [r for r in ev_table(P, threshold=threshold) if r["signal"]]
    if not rows:
        return None
    return max(rows, key=lambda r: r["ev"])