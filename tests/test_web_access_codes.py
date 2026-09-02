"""分账户口令体系的不变量 (2026-08-18 上线)

口令即身份: 213213 全站只读 / 611611 全站可写 / px·llx·phy·xjb·fyf
只见(且只能改)自己名下的线。守住三条:
  1. 口令表结构合法 (pids 都存在、角色合法、不与保留身份混淆)
  2. token 不可伪造/不可跨口令挪用/过期即死
  3. 写权限矩阵: 账户只能写自己的线, ro 什么都写不了
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class _Req:
    """最小 Request 桩: 鉴权函数只用 .cookies"""

    def __init__(self, cookies=None):
        self.cookies = cookies or {}


def _token(ws, code, exp=None):
    exp = exp or int(time.time()) + 3600
    return f"{exp}.c.{ws._code_id(code)}.{ws._view_sign_code(exp, code)}"


def test_access_codes_wellformed():
    from live_config import ACCESS_CODES, PROFILES
    assert ACCESS_CODES, "口令表不能为空"
    for code, spec in ACCESS_CODES.items():
        assert spec["role"] in ("ro", "admin", "acct"), code
        if spec["role"] == "acct":
            assert spec.get("pids"), f"{code} 没配线"
            for pid in spec["pids"]:
                assert pid in PROFILES, f"{code} 指向不存在的线 {pid}"


def test_user_specified_codes_present():
    """2026-08-18 用户指定的口令一个都不能少/映射不能错"""
    from live_config import ACCESS_CODES
    assert ACCESS_CODES["213213"]["role"] == "ro"
    assert ACCESS_CODES["611611"]["role"] == "admin"
    assert set(ACCESS_CODES["px"]["pids"]) == {"steady5w", "aggr2w_px2"}
    assert ACCESS_CODES["llx"]["pids"] == ("aggr5w",)
    assert ACCESS_CODES["phy"]["pids"] == ("aggr2w",)
    assert ACCESS_CODES["xjb"]["pids"] == ("aggr10w",)
    assert ACCESS_CODES["fyf"]["pids"] == ("fyf100w",)


def test_token_roundtrip_roles():
    import web_server as ws
    assert ws._view_role(_Req({ws.VIEW_COOKIE: _token(ws, "px")})) \
        == ("acct", "px")
    assert ws._view_role(_Req({ws.VIEW_COOKIE: _token(ws, "611611")})) \
        == "admin"
    assert ws._view_role(_Req({ws.VIEW_COOKIE: _token(ws, "213213")})) == "ro"
    assert ws._view_role(_Req()) is None


def test_scope_resolution():
    import web_server as ws
    req = _Req({ws.VIEW_COOKIE: _token(ws, "px")})
    assert set(ws._view_scope(req)) == {"steady5w", "aggr2w_px2"}
    # 全站身份不设限
    assert ws._view_scope(_Req({ws.VIEW_COOKIE: _token(ws, "611611")})) is None
    assert ws._view_scope(_Req({ws.VIEW_COOKIE: _token(ws, "213213")})) is None


def test_forged_expired_or_tampered_tokens_rejected():
    import web_server as ws
    exp = int(time.time()) + 3600
    # 拿 llx 的密钥给 px 的 cid 签名 -> 必须失败
    forged = f"{exp}.c.{ws._code_id('px')}.{ws._view_sign_code(exp, 'llx')}"
    assert ws._view_role(_Req({ws.VIEW_COOKIE: forged})) is None
    # 过期 token
    assert ws._view_role(
        _Req({ws.VIEW_COOKIE: _token(ws, "fyf", exp=int(time.time()) - 5)})) \
        is None
    # 签名截断/篡改
    good = _token(ws, "fyf")
    assert ws._view_role(_Req({ws.VIEW_COOKIE: good[:-2] + "zz"})) is None
    # 未知 cid
    bogus = f"{exp}.c.abcdef0123.{'0' * 32}"
    assert ws._view_role(_Req({ws.VIEW_COOKIE: bogus})) is None


def test_write_permission_matrix():
    import web_server as ws
    px = _Req({ws.VIEW_COOKIE: _token(ws, "px")})
    assert ws._write_deny(px, "steady5w") is None
    assert ws._write_deny(px, "aggr2w_px2") is None
    deny = ws._write_deny(px, "aggr5w")          # 别人的线
    assert deny is not None and deny.status_code == 403
    adm = _Req({ws.VIEW_COOKIE: _token(ws, "611611")})
    assert ws._write_deny(adm, "aggr5w") is None  # admin 全可写
    # 未登录/无 ops -> 拒 (401 need_password 老路径)
    assert ws._write_deny(_Req(), "aggr5w") is not None


class _R(_Req):
    def __init__(self, method="GET", q=None, cookies=None):
        super().__init__(cookies)
        self.method = method
        self.query_params = q or {}


def test_acct_interface_narrowing():
    """账户会话摸不到运维页/回测页/默认线接口和非自家写操作;
    看别人的线不拦 (页面有「看全部」切换)"""
    import web_server as ws

    scope = ("aggr5w",)
    assert ws._acct_deny(_R(), "/pro", scope) is not None
    assert ws._acct_deny(_R(), "/api/bt/list", scope) is not None
    assert ws._acct_deny(_R(), "/api/status", scope) is not None
    # 看别人的线是允许的 (只能看, 写在 _write_deny 拦)
    assert ws._acct_deny(_R(q={"profile": "steady5w"}), "/api/today",
                         scope) is None
    assert ws._acct_deny(_R(q={"profile": "aggr5w"}), "/api/today",
                         scope) is None
    assert ws._acct_deny(_R(), "/", scope) is None
    assert ws._acct_deny(_R("POST"), "/api/profile/confirm", scope) is None
    assert ws._acct_deny(_R("POST"), "/api/signal", scope) is not None
    assert ws._acct_deny(_R("POST"), "/api/sync", scope) is not None


def test_effective_scope_all_toggle():
    """?all=1 只放宽"看": 账户会话切到全部线, 写权限矩阵不变"""
    import web_server as ws
    ck = {ws.VIEW_COOKIE: _token(ws, "px")}
    # 默认: 只看自己
    assert set(ws._effective_scope(_R(cookies=ck))) \
        == {"steady5w", "aggr2w_px2"}
    # all=1: 看全部
    assert ws._effective_scope(_R(q={"all": "1"}, cookies=ck)) is None
    # 全站身份永远不设限, all 参数无害空转
    adm = {ws.VIEW_COOKIE: _token(ws, "611611")}
    assert ws._effective_scope(_R(q={"all": "1"}, cookies=adm)) is None
    # 看全部时写别人的线仍被拒
    deny = ws._write_deny(_R(q={"all": "1"}, cookies=ck), "aggr5w")
    assert deny is not None and deny.status_code == 403


def test_profiles_listing_filtered():
    from action_page import list_profiles
    ids_all = {p["id"] for p in list_profiles()}
    assert "steady5w" in ids_all and len(ids_all) >= 11
    ids_px = {p["id"] for p in list_profiles(("steady5w", "aggr2w_px2"))}
    assert ids_px == {"steady5w", "aggr2w_px2"}


def test_resolve_pid_falls_into_scope():
    from action_page import _resolve_pid
    assert _resolve_pid(None, ("fyf100w",)) == "fyf100w"
    assert _resolve_pid("steady5w", ("fyf100w",)) == "fyf100w"
    assert _resolve_pid("aggr2w_px2", ("steady5w", "aggr2w_px2")) \
        == "aggr2w_px2"
    assert _resolve_pid(None, None) == "steady5w"
