"""Lightweight rule-based query parser for DirectMe.

Maps a free-form question (zh / en) into a :class:`QueryIntent`: target
labels, target colors, and four boolean flags (``wants_count``,
``wants_location``, ``wants_reachability``, ``wants_trajectory``).

The parser is deliberately small and extensible. The default
:data:`OBJECT_ALIASES` covers the most common indoor / outdoor categories
that appear in UCS-Bench-like scenes, drawn from the Objects365 taxonomy
plus a few egocentric-specific additions (e.g. trash, bag, key). Callers
needing finer-grained recognition can:

* pass ``extra_aliases={"vending_machine": ["vending machine", "自动售货机"]}``
  to :func:`parse_query`, or
* register them globally via :func:`register_object_aliases`.

For full open-vocabulary intent parsing (CLIP / sentence-transformer label
matching, MLLM-driven structured intent), wrap :func:`parse_query` and
override the ``labels`` field on the returned ``QueryIntent``. The rest of
the retrieval pipeline only consumes the intent dataclass, never the
parser internals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


COLOR_ALIASES: dict[str, list[str]] = {
    "red":    ["red", "红", "红色"],
    "blue":   ["blue", "蓝", "蓝色"],
    "green":  ["green", "绿", "绿色"],
    "yellow": ["yellow", "黄", "黄色"],
    "black":  ["black", "黑", "黑色"],
    "white":  ["white", "白", "白色"],
    "orange": ["orange", "橙", "橙色"],
    "purple": ["purple", "紫", "紫色"],
    "pink":   ["pink", "粉", "粉色"],
    "brown":  ["brown", "棕", "棕色", "褐色"],
    "gray":   ["gray", "grey", "灰", "灰色"],
}


# Default alias table. Covers the most common Objects365 / household / office /
# kitchen / bathroom / outdoor categories likely to appear in UCS-Bench. Not
# exhaustive — extend with `register_object_aliases` for domain-specific labels.
OBJECT_ALIASES: dict[str, list[str]] = {
    # --- drinkware / kitchen small items -----------------------------------
    "cup":        ["cup", "mug", "杯", "杯子", "水杯", "茶杯", "咖啡杯"],
    "bottle":     ["bottle", "瓶子", "水瓶"],
    "bowl":       ["bowl", "碗"],
    "plate":      ["plate", "dish", "盘子", "碟子"],
    "spoon":      ["spoon", "勺子", "汤匙"],
    "fork":       ["fork", "叉子"],
    "knife":      ["knife", "刀", "刀具"],
    "chopsticks": ["chopsticks", "筷子"],
    # --- electronics / personal devices ------------------------------------
    "phone":      ["phone", "mobile", "cellphone", "smartphone", "手机", "电话"],
    "laptop":     ["laptop", "notebook", "笔记本", "笔记本电脑"],
    "tablet":     ["tablet", "ipad", "平板"],
    "tv":         ["tv", "television", "电视", "电视机"],
    "monitor":    ["monitor", "screen", "显示器"],
    "remote":     ["remote", "remote control", "遥控器"],
    "keyboard":   ["keyboard", "键盘"],
    "mouse":      ["mouse", "鼠标"],
    "camera":     ["camera", "相机", "摄像头"],
    "headphones": ["headphones", "headphone", "earphones", "earbuds", "耳机"],
    # --- furniture / fixtures ---------------------------------------------
    "chair":      ["chair", "stool", "椅子", "凳子"],
    "table":      ["table", "desk", "桌子", "餐桌", "书桌"],
    "sofa":       ["sofa", "couch", "沙发"],
    "bed":        ["bed", "床"],
    "cabinet":    ["cabinet", "cupboard", "柜子", "橱柜"],
    "shelf":      ["shelf", "bookshelf", "架子", "书架"],
    # --- kitchen appliances -----------------------------------------------
    "sink":       ["sink", "水槽", "洗手池", "水池"],
    "fridge":     ["fridge", "refrigerator", "冰箱"],
    "oven":       ["oven", "烤箱"],
    "microwave":  ["microwave", "微波炉"],
    "stove":      ["stove", "cooktop", "炉灶", "灶台"],
    "kettle":     ["kettle", "水壶", "热水壶"],
    "toaster":    ["toaster", "面包机", "烤面包机"],
    # --- bathroom ---------------------------------------------------------
    "toilet":     ["toilet", "马桶"],
    "shower":     ["shower", "淋浴", "花洒"],
    "mirror":     ["mirror", "镜子"],
    "towel":      ["towel", "毛巾"],
    # --- doors / openings -------------------------------------------------
    "door":       ["door", "门"],
    "window":     ["window", "窗", "窗户"],
    # --- bags / containers ------------------------------------------------
    "bag":        ["bag", "handbag", "backpack", "包", "袋子", "背包", "手提包"],
    "box":        ["box", "carton", "盒子", "纸箱"],
    "trash":      ["trash", "garbage", "bin", "trash can", "垃圾桶", "垃圾箱"],
    # --- food / consumables -----------------------------------------------
    "toast":      ["toast", "吐司"],
    "bread":      ["bread", "面包"],
    "fruit":      ["fruit", "apple", "banana", "orange", "水果", "苹果", "香蕉"],
    "butter":     ["butter", "黄油", "奶油"],
    # --- small personal items ---------------------------------------------
    "key":        ["key", "keys", "钥匙"],
    "pen":        ["pen", "pencil", "笔", "铅笔"],
    "book":       ["book", "magazine", "书", "书本", "杂志"],
    "wallet":     ["wallet", "purse", "钱包"],
    # --- people / clothing ------------------------------------------------
    "person":     ["person", "people", "human", "人"],
    "shoe":       ["shoe", "shoes", "sneaker", "鞋", "鞋子"],
    "hat":        ["hat", "cap", "帽子"],
    # --- outdoor / vehicles -----------------------------------------------
    "car":        ["car", "vehicle", "汽车", "车"],
    "bicycle":    ["bike", "bicycle", "自行车", "单车"],
    "bus":        ["bus", "公交车", "公交"],
    # --- 改动 B（修正版）：以节点实际 semantic_label 为 KEY ─────────────────
    # 关键原则：_score_node 检查的是 node.semantic_label.lower() 是否
    #           包含 intent_label（即 KEY）；alias 列表是用户查询中会出现的词。
    #
    # 当前感知端（YOLO-World）把"墙上的纸/海报/照片"都识别成 Picture/Frame。
    # 所以 KEY = "picture"（能命中 "picture/frame"），
    # alias 包含 paper/sheet/poster 等用户查询词。
    "picture":    ["picture", "pictures", "photo", "photos", "photograph",
                   "frame", "frames", "picture frame",
                   # 感知误识：墙上的纸被识别为 Picture/Frame
                   "paper", "papers", "sheet", "sheets", "page", "pages",
                   "note", "notes", "poster", "posters", "print",
                   "artwork", "painting",
                   "图片", "照片", "画框", "相框", "纸", "纸张",
                   "便签", "文件", "海报", "画"],
    # "Blackboard/Whiteboard" 节点
    "blackboard": ["blackboard", "whiteboard", "chalkboard", "board",
                   "black board", "white board",
                   "黑板", "白板", "写字板"],
    # "Trash bin Can" 节点（已有 trash key，补全 "garbage can" 等组合词）
    "trash_can":  ["trash can", "garbage can", "trash bin", "rubbish bin",
                   "dustbin", "废纸篓"],
    # "Washing Machine/Drying Machine" 节点
    "washing":    ["washing machine", "washing", "dryer", "laundry machine",
                   "洗衣机", "烘干机"],
    # "Barrel/bucket" 节点
    "barrel":     ["barrel", "bucket", "pail", "tub",
                   "桶", "水桶", "盆"],
    # 推车 / 购物车
    "trolley":    ["trolley", "trolleys", "cart", "carts",
                   "shopping cart", "dolly",
                   "推车", "小推车", "手推车", "购物车"],
    # 手提包（已有 bag key，补精确词）
    "handbag":    ["handbag", "handbags", "tote", "clutch",
                   "手提包", "手提袋", "提包"],
    # 打印机
    "printer":    ["printer", "printers", "打印机"],
    # 电源插座 → "Power outlet" 节点
    "outlet":     ["power outlet", "outlet", "socket", "plug",
                   "插座", "电源插座"],
}


# Room / place names live in their OWN table because they are not objects in
# the scene graph — they are surfaced via ``place_visit_timeline`` and via
# ``EntityNode.attributes["scene_tag"]``. Keeping them out of
# OBJECT_ALIASES prevents a question like "客厅那个红杯子在哪？" from
# accidentally adding ``living_room`` to ``QueryIntent.labels`` and skewing
# retrieval.
ROOM_ALIASES: dict[str, list[str]] = {
    "kitchen":      ["kitchen", "厨房"],
    "living_room":  ["living room", "客厅", "起居室"],
    "bedroom":      ["bedroom", "卧室"],
    "bathroom":     ["bathroom", "restroom", "卫生间", "浴室", "厕所"],
    "office":       ["office", "办公室"],
    "hallway":      ["hallway", "corridor", "走廊", "过道"],
    "dining_room":  ["dining room", "餐厅"],
}


def register_object_aliases(extra: dict[str, list[str]]) -> None:
    """Globally extend the default alias table.

    Useful for domain deployments (factory, lab, classroom). Aliases passed
    here merge into :data:`OBJECT_ALIASES` and persist for the lifetime of
    the process.
    """
    for label, aliases in extra.items():
        OBJECT_ALIASES.setdefault(label, []).extend(
            a for a in aliases if a not in OBJECT_ALIASES.get(label, [])
        )


@dataclass
class QueryIntent:
    raw_query: str
    labels: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    rooms: list[str] = field(default_factory=list)
    wants_count: bool = False
    wants_location: bool = False
    wants_reachability: bool = False
    wants_trajectory: bool = False
    language: str = "zh"


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def _contains_term(text: str, term: str) -> bool:
    """Robust alias matching.

    Chinese aliases are matched by substring. English aliases use token
    boundaries so short labels such as ``cup`` do not fire on ``cupboard`` or
    ``car`` on ``cart``. Multi-word aliases (``remote control``) are matched
    with flexible whitespace.
    """
    t = term.lower().strip()
    if not t:
        return False
    if _has_cjk(t):
        return t in text
    pieces = [re.escape(part) for part in re.split(r"\s+", t) if part]
    if not pieces:
        return False
    pattern = r"(?<![a-z0-9])" + r"\s+".join(pieces) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def parse_query(
    query: str,
    language: str | None = None,
    extra_aliases: dict[str, list[str]] | None = None,
) -> QueryIntent:
    """Parse a free-form question into a structured :class:`QueryIntent`.

    Args:
        query: the raw user question, in zh or en.
        language: ``"zh"`` / ``"en"`` to force, or ``None`` to auto-detect.
        extra_aliases: per-call additions to the alias table; merged with
            the global :data:`OBJECT_ALIASES` for this call only. Useful when
            you want to add scene-specific vocabulary without polluting the
            global registry.
    """
    q = query.lower()

    aliases = OBJECT_ALIASES
    if extra_aliases:
        aliases = {**OBJECT_ALIASES}
        for label, more in extra_aliases.items():
            aliases[label] = list(dict.fromkeys((aliases.get(label, []) + list(more))))

    labels = [label for label, alias_list in aliases.items() if _contains_any(q, alias_list)]
    colors = [color for color, alias_list in COLOR_ALIASES.items() if _contains_any(q, alias_list)]
    rooms = [room for room, alias_list in ROOM_ALIASES.items() if _contains_any(q, alias_list)]

    wants_count = bool(re.search(r"\bhow many\b|\bcount\b|几个|多少|数量", q))
    wants_location = bool(re.search(r"\bwhere\b|\blocation\b|\brelative\b|在哪|哪里|方位|身边|附近", q))
    wants_reachability = bool(
        re.search(
            r"\breach\b|\bwithin reach\b|够得着|够得到|能拿到|拿得到|能不能拿到|能否到达|"
            r"伸手可及|手边|是否可达|在我手边",
            q,
        )
    )
    wants_trajectory = bool(
        re.search(
            r"\bpath\b|\btrajectory\b|\bwhere did i\b|\bhave i been\b|\broute\b|"
            r"\brooms?\b.*\b(visited|been to)\b|"
            r"路径|轨迹|走过|去过|经过|移动到|搬到|从.*到|路线|哪些房间",
            q,
        )
    )

    if language is None:
        language = "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"

    return QueryIntent(
        raw_query=query,
        labels=labels,
        colors=colors,
        rooms=rooms,
        wants_count=wants_count,
        wants_location=wants_location,
        wants_reachability=wants_reachability,
        wants_trajectory=wants_trajectory,
        language=language,
    )
