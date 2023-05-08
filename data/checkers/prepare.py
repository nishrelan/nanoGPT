

rows = 'abcdefgh'
cols = '12345678'


def chartoint(pos):
    if len(pos) != 2:
        return -1
    if pos[0] not in rows or pos[1] not in cols:
        return -1
    return rows.index(pos[0]) * 8 + cols.index(pos[1])
    

def inttochar(pos):
    if not 0 <= pos <= 63:
        return -1
    l = pos // 8
    c = pos % 8
    return rows[l] + cols[c]
