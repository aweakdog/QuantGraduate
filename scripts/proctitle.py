"""进程题名低调化 (2026-08-22)

背景: 实验室三台机器人人有 root, htop/ps 里默认显示完整命令行
(路径 quant-strategy + 脚本名 wf_v35/live_signal 一眼暴露研究内容)。
setproctitle 会改写 /proc/<pid>/cmdline 的显示, 把整条命令行换成一个
不显眼的标题 —— 只影响"别人顺眼看到什么", 不影响 sys.argv/指纹/功能。

约定的题名(改这里前先想好 web_server._MACH_CMD 里的 pgrep 是按题名数进程的):
    mltask/srv     web_server 常驻
    mltask/worker  wf_v35 回测(实验时一跑几十个, 最显眼)
    mltask/grid    eval_grid 编排
    mltask/etl     daily_rebuild 晚间链
    mltask/task    live_signal
    mltask/fetch   update_kline_akshare
    mltask/feat    tick_micro_features (040)

setproctitle 未安装时静默跳过: 隐私是软需求, 绝不能因它挂掉 17:30 链。
"""


def lowkey(title):
    try:
        import setproctitle
        setproctitle.setproctitle(title)
    except Exception:
        pass
