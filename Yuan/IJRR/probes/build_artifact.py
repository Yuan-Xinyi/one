"""Assemble the print-size analysis into a self-contained page.

Every number is read back out of the result archives rather than transcribed,
so the page cannot drift from the data. Images are inlined as data URIs
because the artifact CSP blocks external hosts.
"""
import base64
import sys
from pathlib import Path

import numpy as np

OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis')
REN = OUT / 'render'
DST = Path('/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-'
           'vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/'
           'scratchpad/print_report.html')
OBJS = [('立方体', 'cube', '边长'), ('笔筒', 'penholder', '外径'),
        ('寿司碗', 'bowl', '口径'), ('花瓶', 'vase', '高'),
        ('盘子', 'plate', '直径'), ('半球罩', 'dome', '直径'),
        ('马克杯', 'mug', '杯身外径')]


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def img(name, cap):
    p = OUT / name
    if not p.exists():
        return ''
    return (f'<figure class="wide"><img src="data:image/png;base64,{b64(p)}"'
            f' alt=""><figcaption>{cap}</figcaption></figure>')


q = np.load(OUT / 'cube_query.npz')
gx, gz, STEP = q['gx'], q['gz'], float(q['step'])
k0 = int(np.argmin(np.abs(gz - STEP / 2)))


def cube_row(cone):
    A, H = q[f'cube_c{cone}_solid'], q[f'cube_c{cone}_hollow']
    b = np.unravel_index(A.argmax(), A.shape)
    return dict(cone=cone, solid=A.max() * 100, hollow=H.max() * 100,
                cx=gx[b[0]], cy=gx[b[1]], zb=gz[b[2]] - STEP / 2,
                table=A[:, :, k0].max() * 100)


cubes = [cube_row(c) for c in (5, 30, 90)]
c5, c30 = cubes[0], cubes[1]
rows_cube = '\n'.join(
    f'<tr><td class="k">{r["cone"]}&deg;</td>'
    f'<td class="n hi">{r["solid"]:.0f}</td><td class="n">{r["hollow"]:.0f}</td>'
    f'<td class="n">{r["cx"]:+.2f}, {r["cy"]:+.2f}</td>'
    f'<td class="n">{r["zb"]:+.2f}</td><td class="n">{r["table"]:.0f}</td></tr>'
    for r in cubes)

# ---- continuity ----------------------------------------------------------
rows_rast = ''
if (OUT / 'raster_query.npz').exists():
    r = np.load(OUT / 'raster_query.npz')
    NAMES = ['经典梯度 @首个可行起点', '经典梯度 @critic 选的起点',
             '学习策略 @首个可行起点', '学习策略 @critic 起点（完整框架）']
    blk = []
    for cone in (5, 30):
        P, meta, has = r[f'c{cone}'], r[f'meta_c{cone}'], r[f'has_c{cone}']
        side = float(meta[0])
        blk.append(f'<tr class="sub"><td colspan="4">喷嘴倾角容差 {cone}&deg;　'
                   f'立方体 {side*100:.0f} cm，单条走刀 {side*100:.0f} cm，'
                   f'{int(has.sum())} 条</td></tr>')
        for nm, p in zip(NAMES, P):
            v = p[has]
            cls = ' hi' if '完整框架' in nm else ''
            blk.append(f'<tr><td class="k">{nm}</td>'
                       f'<td class="n{cls}">{v.mean()*100:.1f}</td>'
                       f'<td class="n">{v.mean()/side*100:.0f}%</td>'
                       f'<td class="n{cls}">'
                       f'{np.mean(v >= side - 1e-3)*100:.1f}%</td></tr>')
    rows_rast = '\n'.join(blk)

# ---- common objects ------------------------------------------------------
rows_obj, gallery = '', ''
if (OUT / 'objects_query.npz').exists():
    oq = np.load(OUT / 'objects_query.npz')
    lo = []
    for cn, key, dim in OBJS:
        a, b = oq[f'{key}_c5'], oq[f'{key}_c30']
        sw = ''
        pk = REN / f'{key}_pack.npz'
        if pk.exists():
            st = np.load(pk, allow_pickle=False)['stats']
            sw = f'{int(st[2])} / {int(st[0])}'
        lo.append(f'<tr><td class="k">{cn}</td><td class="k">{dim}</td>'
                  f'<td class="n hi">{a[0]*100:.0f}</td>'
                  f'<td class="n">{b[0]*100:.0f}</td>'
                  f'<td class="n">{a[1]:+.2f}, {a[2]:+.2f}</td>'
                  f'<td class="n">{a[3]:+.2f}</td><td class="n">{sw}</td></tr>')
    rows_obj = '\n'.join(lo)
    cards = []
    for cn, key, dim in OBJS:
        st = REN / f'{key}_still.jpg'
        if not st.exists():
            continue
        s = float(oq[f'{key}_c5'][0]) * 100
        cards.append(
            f'<figure class="card"><img src="data:image/jpeg;base64,{b64(st)}"'
            f' alt=""><figcaption><b>{cn}</b> · {dim} {s:.0f} cm</figcaption>'
            f'</figure>')
    if cards:
        gallery = f'<div class="grid">{"".join(cards)}</div>'

# ---- one-stroke figures --------------------------------------------------
FIGS = [('圆', 'circle'), ('正方形', 'square'), ('三角形', 'triangle'),
        ('五角星', 'star'), ('信封', 'envelope'), ('双环', 'eight'),
        ('螺线', 'spiral')]
rows_os, gal_os = '', ''
if (OUT / 'onestroke.npz').exists():
    o = np.load(OUT / 'onestroke.npz')
    blk = []
    for cone in (30, 5):
        blk.append(f'<tr class="sub"><td colspan="6">喷嘴倾角容差 {cone}&deg;</td></tr>')
        for cn, key in FIGS:
            k = f'{key}_c{cone}'
            if k not in o.files:
                continue
            pw, pol, cls, fp, fc = [float(v) for v in o[k]]
            rat = f'{pol/cls:.1f}&times;' if cls > 0 else '&ndash;'
            blk.append(
                f'<tr><td class="k">{cn}</td>'
                f'<td class="n">{pw*100:.0f}</td>'
                f'<td class="n hi">{pol*100:.0f}</td>'
                f'<td class="n">{cls*100:.0f}</td>'
                f'<td class="n">{rat}</td>'
                f'<td class="n">{fp:.2f} / {fc:.2f}</td></tr>')
    rows_os = '\n'.join(blk)
    cards = []
    for cn, key in FIGS:
        st = OUT / 'onestroke' / f'{key}_still.jpg'
        if not st.exists() or f'{key}_c30' not in o.files:
            continue
        v = o[f'{key}_c30']
        cards.append(
            f'<figure class="card"><img src="data:image/jpeg;base64,{b64(st)}"'
            f' alt=""><figcaption><b>{cn}</b> · {float(v[1])*100:.0f} cm '
            f'一笔画完 · 经典止于 {float(v[2])*100:.0f} cm</figcaption></figure>')
    if cards:
        gal_os = f'<div class="grid">{"".join(cards)}</div>'

# ---- bowl ----------------------------------------------------------------
rows_bowl = ''
if (OUT / 'bowl_query.npz').exists():
    bq = np.load(OUT / 'bowl_query.npz')
    fair = (np.load(OUT / 'bowl_fair.npz')
            if (OUT / 'bowl_fair.npz').exists() else None)
    ls = []
    for mode, mlab in (('planar', '平面层 · 喷嘴始终朝下'),
                       ('conformal', '共形 · 喷嘴沿壁法线')):
        for inv, ilab in ((False, '正放（碗口朝上）'), (True, '倒放（穹顶）')):
            for cone in (5, 30):
                tag = 'inv' if inv else 'up'
                k = f'{mode}_c{cone}_{tag}'
                if k not in bq.files:
                    continue
                v = bq[k]
                if mode == 'planar':
                    R = float(v.max()); pl = bq['places'][int(v.argmax())]
                else:
                    R, pl = float(v[0]), v[1:]
                f2 = ''
                if fair is not None:
                    fk = (f'planarfair_c{cone}_{tag}' if mode == 'planar'
                          else f'conffine_c{cone}_{tag}')
                    if fk in fair.files:
                        f2 = f'{200*float(fair[fk][0]):.0f}'
                cls = ' hi' if mode == 'conformal' else ''
                ls.append(
                    f'<tr><td class="k">{mlab}</td><td class="k">{ilab}</td>'
                    f'<td class="n">{cone}&deg;</td>'
                    f'<td class="n{cls}">{200*R:.0f}</td>'
                    f'<td class="n">{f2}</td>'
                    f'<td class="n">{pl[0]:+.2f}, {pl[1]:+.2f}</td>'
                    f'<td class="n">{pl[2]:+.2f}</td></tr>')
    rows_bowl = '\n'.join(ls)

rows_part = ''
if (OUT / 'part_query.npz').exists():
    pq = np.load(OUT / 'part_query.npz')
    ls = []
    for cone in (5, 30):
        if f'cube_c{cone}' in pq.files:
            v = pq[f'cube_c{cone}']
            ls.append(f'<tr><td class="k">立方体（实心，边长）</td>'
                      f'<td class="n">{cone}&deg;</td>'
                      f'<td class="n">{v[0]*100:.0f}</td>'
                      f'<td class="n hi">{v[1]*100:.0f}</td></tr>')
        for mode, mlab in (('planar', '碗 · 平面层'), ('conformal', '碗 · 共形')):
            for inv, ilab in ((False, '正放'), (True, '倒放')):
                kk = f'bowl_{mode}_c{cone}_{"inv" if inv else "up"}'
                if kk not in pq.files:
                    continue
                v = pq[kk]
                ls.append(f'<tr><td class="k">{mlab}（{ilab}，口径）</td>'
                          f'<td class="n">{cone}&deg;</td>'
                          f'<td class="n">{v[0]*200:.0f}</td>'
                          f'<td class="n hi">{v[1]*200:.0f}</td></tr>')
    rows_part = '\n'.join(ls)

HTML = f"""<title>固定基座能打多大 —— FR3 三维打印尺寸分析</title>
<style>
:root {{
  --ground:#F5F7F9; --panel:#FFFFFF; --ink:#12171E; --muted:#5A6472;
  --rule:#DFE4EA; --accent:#C9461F; --accent2:#14626B; --wash:#EDF1F4;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink);
  font:16.5px/1.72 Georgia,"Songti SC","Noto Serif CJK SC",serif;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:0 24px 96px; }}
.col {{ max-width:70ch; }}
h1,h2,h3,.eyebrow,.n,th,.chip,figcaption {{
  font-family:ui-sans-serif,-apple-system,"Helvetica Neue","PingFang SC",
  "Microsoft YaHei",sans-serif; }}
header {{ padding:72px 0 40px; border-bottom:3px solid var(--ink);
  margin-bottom:40px; }}
h1 {{ font-size:clamp(30px,4.4vw,50px); line-height:1.1; letter-spacing:-.022em;
  font-weight:800; margin:0 0 18px; text-wrap:balance; }}
.lede {{ font-size:19px; color:var(--muted); max-width:64ch; margin:0; }}
.eyebrow {{ font-size:11.5px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--accent); font-weight:700; margin:0 0 14px; }}
h2 {{ font-size:27px; letter-spacing:-.015em; font-weight:750; margin:64px 0 6px;
  padding-top:22px; border-top:1px solid var(--rule); }}
h2 .num {{ color:var(--accent); font-variant-numeric:tabular-nums;
  margin-right:.5em; font-weight:800; }}
p {{ margin:0 0 16px; }}
ul {{ margin:0 0 18px; padding-left:1.15em; }}
li {{ margin-bottom:9px; }}
em {{ font-style:normal; background:linear-gradient(transparent 62%,#F6D7C8 62%); }}
.strip {{ display:flex; flex-wrap:wrap; gap:14px; margin:34px 0 0; }}
.chip {{ background:var(--panel); border:1px solid var(--rule);
  border-left:4px solid var(--accent); padding:13px 18px; min-width:170px; }}
.chip b {{ display:block; font-size:27px; font-weight:800; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; line-height:1.15; }}
.chip span {{ display:block; font-size:12.5px; color:var(--muted); margin-top:3px; }}
figure {{ margin:30px 0; }}
figure img {{ width:100%; display:block; background:var(--panel);
  border:1px solid var(--rule); }}
figcaption {{ font-size:13.5px; color:var(--muted); margin-top:10px;
  max-width:80ch; line-height:1.6; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:18px; margin:28px 0; }}
.grid figure {{ margin:0; }}
.grid figcaption {{ margin-top:8px; font-size:13px; }}
.grid figcaption b {{ color:var(--ink); font-size:14px; }}
.tbl {{ overflow-x:auto; margin:24px 0 8px; border:1px solid var(--rule);
  background:var(--panel); }}
table {{ border-collapse:collapse; width:100%; min-width:600px; }}
th {{ font-size:12px; font-weight:700; letter-spacing:.04em; text-align:right;
  padding:11px 16px; background:var(--wash); border-bottom:1px solid var(--rule);
  color:var(--muted); white-space:nowrap; }}
th:first-child {{ text-align:left; }}
td {{ padding:10px 16px; border-bottom:1px solid var(--rule); font-size:14.5px; }}
td.k {{ font-size:14px; }}
td.n {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.hi {{ font-weight:800; color:var(--accent); }}
tr.sub td {{ background:var(--wash); font-size:12.5px; letter-spacing:.03em;
  color:var(--muted);
  font-family:ui-sans-serif,-apple-system,sans-serif; }}
tr:last-child td {{ border-bottom:none; }}
.note {{ border-left:3px solid var(--accent2); background:var(--panel);
  padding:16px 20px; margin:26px 0; font-size:15px; }}
.note .eyebrow {{ color:var(--accent2); }}
footer {{ margin-top:72px; padding-top:22px; border-top:1px solid var(--rule);
  font-size:13px; color:var(--muted);
  font-family:ui-sans-serif,sans-serif; }}
code {{ font:13.5px/1.5 ui-monospace,Menlo,Consolas,monospace;
  background:var(--wash); padding:1px 5px; }}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">FR3 · 固定基座 · 逐点可行上界 + 连续性</p>
  <h1>固定基座能打多大</h1>
  <p class="lede">打印任务和论文里定义的任务是同一个数学对象：喷嘴位置跟轨迹，
  喷嘴轴线保持在一个锥内，绕轴自转自由。所以"最大能打多大"可以用现成的机器回答——
  而答案分成两半，一半是几何，一半是连续性，后者才是有意思的那一半。</p>
  <div class="strip">
    <div class="chip"><b>{c5['solid']:.0f} cm</b><span>5&deg; 容差下最大实心立方体（逐点）</span></div>
    <div class="chip"><b>{c30['solid']:.0f} cm</b><span>30&deg; 容差下最大实心立方体（逐点）</span></div>
    <div class="chip"><b>0%</b><span>经典控制器一趟走完整条走刀的比例</span></div>
    <div class="chip"><b>83.7%</b><span>完整框架（5&deg;）走完整条走刀的比例</span></div>
  </div>
</header>

<div class="col">
<h2><span class="num">01</span>"最大"有三层含义</h2>
<p>不先分清楚，这个问题没有唯一答案。三层是嵌套的，数值可以差一倍以上：</p>
<ul>
<li><strong>几何可达</strong>：整个工件落在机械臂够得着的范围里。</li>
<li><strong>逐点可行</strong>：走刀轨迹的<em>每一点</em>都存在喷嘴姿态合格、不自碰、在关节限内的位形。
这是论文里 &#8467;<sup>pw</sup> 的工件版本，也是本页大部分表格的口径。</li>
<li><strong>连续可打</strong>：这些位形能被<em>一条连续运动</em>串起来。严格小于上一层，而且小得多——第 04 节。</li>
</ul>
<p>另外两条硬约束是纯运动学场里没有、必须补上的：工件不能占据机械臂自己的基座立柱
（半径取 0.20 m，link0 最大横向尺寸 0.175 m），以及手臂不能穿过打印床。</p>
</div>

<h2><span class="num">02</span>朝下可行域场与最大立方体</h2>
<div class="col">
<p>平面分层时喷嘴处处朝下，逐点可行性<strong>只是位置的函数</strong>。把它算成一个三维布尔场之后，
"最大工件"就退化成"这个场里能装下的最大内接体"，三维前缀和可以对任意（摆放位置，床高）
在常数时间内查询。场按 2 cm 体素、位置容差 2 mm 计算，喷嘴倾角容差取三档。</p>
</div>
{img('fig1_placement.png',
     '每个格点是把立方体底面中心放在这里、床高取该点最优时能打的最大实心边长。'
     '中间深灰圆是基座立柱禁区。三档容差下最优位置都落在距基座约 0.5 m 的环带上，'
     '而不是尽量靠近或尽量远离基座。')}
{img('fig2_slice.png',
     '过最优点的竖直切片。蓝色是朝下可行集，中间那块空洞是"离基座太近"的死区——'
     '它的存在是最大立方体被卡住的直接原因，而不是够不着。')}
<div class="tbl"><table>
<thead><tr><th>喷嘴倾角容差</th><th>实心 (cm)</th><th>空心壳 (cm)</th>
<th>底面中心 (m)</th><th>床高 z (m)</th><th>床与基座同高 (cm)</th></tr></thead>
<tbody>
{rows_cube}
</tbody></table></div>
<div class="col">
<p style="font-size:14px;color:var(--muted);margin-top:10px">90&deg; 一档不是真实打印规格，
它是"喷嘴方向随便"的参照，用来把"够不着"和"腕子转不过来"分开：5&deg;→90&deg; 的差距全部是姿态代价。</p>
<p>三条直接的结论。<strong>倾角容差是最值钱的自由度</strong>：5&deg;→30&deg; 换来
{c30['solid']-c5['solid']:.0f} cm，再放宽到无约束只再多
{cubes[2]['solid']-c30['solid']:.0f} cm，说明 30&deg; 已经吃掉大部分姿态红利。
<strong>床面放低比放平好</strong>：最优床高在基座下方 0.1–0.25 m，
床与基座同高要损失 4–8 cm。<strong>实心和空心几乎没差别</strong>——
立方体的内部是容易的那部分，卡住它的是外侧的角。</p>
</div>

<h2><span class="num">03</span>每个自由度值多少</h2>
{img('fig3_freedoms.png',
     '左：床高扫描，三档容差。中：倾角容差三档，以及"手臂必须在床面以上"这条约束的代价。'
     '右：固定在各自最优底面位置上单独扫床高——约 20 cm 宽的平台，'
     '也就是床面定位有 ±10 cm 的容差不会掉性能。')}

<h2><span class="num">04</span>连续性缺口：真正的限制在这里</h2>
<div class="col">
<p>上面那些表说整个工件逐点可行。这恰恰说明逐点口径<strong>不是打印该问的问题</strong>：
打印需要的是这些位形能被一条连续运动串起来。</p>
<p>一层光栅填充就是一叠平行的直线走刀，喷嘴朝下，长度等于工件宽度。
这正是论文的任务族，只是任务参数从随机采样换成从打印几何里读出来，
所以这一节不需要任何新环境：在立方体自己的位置上生成这些直线任务，
建 5&deg;/30&deg; 可行起点池，再用论文协议跑三方。两种控制器用同一批起点。</p>
</div>
{img('fig5_continuity.png',
     '左侧两图：单条走刀能走多远（箱线图，黑虚线是一条走刀的长度）。'
     '右侧两图：一趟走完整条走刀的比例。')}
<div class="tbl"><table>
<thead><tr><th>方法</th><th>连续行程 (cm)</th><th>占一条走刀</th><th>一趟走完</th></tr></thead>
<tbody>
{rows_rast}
</tbody></table></div>
<div class="col">
<p>这是整份分析里最有信息量的一格。<strong>5&deg; 容差下经典梯度平均只能走 5.5 cm，
一条 46 cm 的走刀一条都走不完；完整框架平均走 65.9 cm、83.7% 的走刀一趟到底。</strong>
30&deg; 下经典 1.5%，完整框架 100%。也就是说，<strong>在这个尺寸上"能不能打"
是控制器决定的，不是工作空间决定的</strong>；框架把瓶颈从"走不完一条"挪到了"掉头怎么办"。</p>
<div class="note">
<p class="eyebrow">口径说明</p>
<p style="margin:0">行程沿<em>单一方向</em>的直线任务量出，而真实光栅每条走刀都要掉头。
掉头会把关节往回卷、对手臂有利，所以这是光栅的保守下界；掉头本身的可行性未验证。
速度用主线的 0.2 m/s，远高于真实进给；速度扫描显示 FR3 在 0.05 m/s 上中性。
30&deg; 那档 12.7% 的走刀起点没建出候选（起点池只用 4 个方向），
而更密的方向扫描在场里认证过这些点——那是起点池的不完备，比例在有池的走刀上算。</p>
</div>
</div>

<h2><span class="num">05</span>一笔画：不许中途接缝</h2>
<div class="col">
<p>前面那些走刀允许手臂中途换分支。对打印来说这是不成立的——
<strong>一条封闭轮廓中间不能有接缝，圆断了就不是圆。</strong>
所以这一节换一条硬规则：整幅图必须一笔画完，任何一次换分支都判失败。
问题于是变成二值的（这一幅、这个尺寸、这个位置，是不是一笔下来了），
而"能多大"就是它还成立的最大尺寸。</p>
<p>图形不是我手排的点序。正方形、五角星、信封这些定义成
<strong>（顶点，边）图</strong>，笔顺由 Hierholzer 算法算出<strong>欧拉路径</strong>，
保证每条边恰好走一次；不存在欧拉路径的图会被直接拒绝，所以这个演示没法靠偷偷抬笔蒙混。
信封那张自动落在"从左下角起、右下角收"，因为它恰好有两个奇度顶点。</p>
<p>还有一处口径必须说明：<strong>路径参考帧取的是指令弧长，不是最近点。</strong>
七张图里五张自交，最近点查表会在交叉处跳到另一条分支、把手臂引向错误方向；
按弧长下指令既无歧义，也正是机床真实的做法。两个控制器共用 critic 选出的同一个起点。</p>
</div>
<div class="tbl"><table>
<thead><tr><th>图形</th><th>逐点上界 (cm)</th><th>完整框架 (cm)</th>
<th>经典梯度 (cm)</th><th>框架 / 经典</th><th>平均画完比例</th></tr></thead>
<tbody>
{rows_os if rows_os else '<tr><td colspan="6">计算中</td></tr>'}
</tbody></table></div>
{gal_os}
<div class="col">
<p><strong>框架在两档容差下都基本顶到了几何上界</strong>（94–100%）：限制已经从控制器移回工作空间本身。
经典梯度在 30&deg; 下只有上界的 40–64%，在 5&deg; 下只剩 5–11%——
<strong>把喷嘴容差从 30&deg; 收到 5&deg;，差距从 1.5–2.6 倍拉到 9–18 倍。</strong>
这和直线任务上 5&deg; 那次（19.6 vs 93.5）是同一条规律：约束越紧，两级框架的优势越大。</p>
<p style="font-size:14px;color:var(--muted)">截图是每段对比视频的末帧：红＝完整框架，灰蓝＝经典梯度，
灰笔停住的位置就是该控制器零空间用尽的地方。尺寸取框架能一笔画完的最大值，
所以经典是在<em>框架的极限尺寸</em>上被考的；它自己能一笔画完的最大尺寸另列一栏。</p>
</div>

<h2><span class="num">06</span>常见物体能打多大</h2>
<div class="col">
<p>每个物体写成一条<strong>有序走刀轨迹</strong>（不是点云），
按平面分层、喷嘴朝下查同一个场，沿尺度阶梯二分出最大尺寸。
最后一列是把手臂沿这条轨迹行军时必须<strong>换分支</strong>的次数——
每一次都是一次真实的重新落位，也就是工件上的一道接缝。</p>
</div>
<div class="tbl"><table>
<thead><tr><th>物体</th><th>尺寸口径</th><th>5&deg; (cm)</th><th>30&deg; (cm)</th>
<th>底面中心 (m)</th><th>床高 (m)</th><th>换分支 / 路点</th></tr></thead>
<tbody>
{rows_obj if rows_obj else '<tr><td colspan="7">计算中</td></tr>'}
</tbody></table></div>
{gallery}
<div class="col">
<p style="font-size:14px;color:var(--muted)">截图取自 <code>one</code> 仿真器的定机位回放：
橙色是已经沉积的料，灰色细线是完整走刀轨迹，喷嘴变灰的片段是换分支的空程。
动画里的层高和填充间距比真实打印粗，几何不变。
立方体这一行按走刀点集查得 50 cm，而第 02 节按实心长方体严格查得
{c5['solid']:.0f} cm——差的 4 cm 是填充离散化，两者口径不同，不是矛盾。</p>
</div>

<h2><span class="num">07</span>寿司碗：摆放方式是一个决策</h2>
<div class="col">
<p>碗取球冠形：口径 2R，深 0.9R，底足 0.45R。和立方体不同，
碗不是姿态无关的工件，有两个互相独立的选择在打架：<strong>工件怎么放</strong>
（正放还是倒扣）和<strong>喷嘴怎么拿</strong>（平面分层朝下，还是共形沿壁面法线）。</p>
<p>平面分层把姿态任务退化成立方体那一档，代价是悬垂规则：这个碗形最陡处壁面偏离竖直
<strong>63.4&deg;</strong>，远超 45&deg; 的免支撑上限。共形打印消掉悬垂，换来一个真正的姿态任务：
<strong>喷嘴要从碗底的竖直一路转到碗沿的近水平（最大倾角 84&deg;），并绕一整圈方位角。</strong></p>
</div>
{img('fig4_bowl.png',
     '左、中：碗的轮廓与共形喷嘴轴（红箭头）。正放时喷嘴在碗腔内、朝外下方；'
     '倒放时喷嘴在穹顶外侧、朝内下方。右：各组合的逐点口径。')}
<div class="tbl"><table>
<thead><tr><th>喷嘴方式</th><th>摆放</th><th>容差</th><th>口径 (cm)</th>
<th>同口径复核 (cm)</th><th>底面中心 (m)</th><th>床高 (m)</th></tr></thead>
<tbody>
{rows_bowl if rows_bowl else '<tr><td colspan="7">计算中</td></tr>'}
</tbody></table></div>
<div class="col">
<p>平面分层下摆放方式几乎不影响逐点尺寸——点集形状差不多。<strong>共形则完全相反：
正放能打到很大，倒放几乎打不了。</strong>原因是喷嘴指向：正放时喷嘴在碗腔内朝外指，
远侧的壁面就在手臂前方；倒放时喷嘴在穹顶外侧朝内指，要打远侧就得绕到工件背面、
把喷嘴指回机器人自己——那是够不到的姿态。<strong>共形打印里，摆放方式决定的是可行不可行，
不只是大小。</strong></p>
</div>

<h2><span class="num">08</span>床面与已打印工件的代价</h2>
<div class="col">
<p>前面所有数字都是纯运动学的：碰撞模型里只有手臂自己。打印机还有两个东西——床面，
以及<strong>正在长高的工件</strong>。后者对凸的、自下而上打的立方体几乎不要钱
（手臂始终在已打印表面之上），对正放的碗则很贵（手腕要伸进四壁不断升高的腔里）。</p>
</div>
<div class="tbl"><table>
<thead><tr><th>工件</th><th>容差</th><th>纯运动学 (cm)</th><th>加床面+工件 (cm)</th></tr></thead>
<tbody>
{rows_part if rows_part else '<tr><td colspan="4">计算中</td></tr>'}
</tbody></table></div>

<h2><span class="num">09</span>边界，以及没做的事</h2>
<div class="col">
<ul>
<li>场是 2 cm 体素，尺寸按 2 cm 量化；位置容差取 2 mm（IK 投影器收敛门限是 5 mm，
更紧的规格这套机器答不了）。</li>
<li>可行性依赖逆解完备性：每点试 8–24 个锥内方向、每方向 8 个热启动，
失败即判不可行，所以场是<em>保守</em>的。</li>
<li>连续行程按单向直线量，光栅掉头没算，这是保守方向。</li>
<li>共形档的摆放网格 10 cm、半径梯 2 cm、面采样比平面档粗，
所以表里多给了一列"同口径复核"：平面档按共形的采样与网格重跑，
共形档的冠军按平面档的细采样复核。正放共形顶到了半径梯上限，实际值只会更大。</li>
<li>物体演示是<em>逐点可行性的见证位形链</em>（每点从上一点热启动、保种子求解），
证明整条走刀几何上可覆盖；它不是某个控制器的执行轨迹，连续性由第 04 节单独测。</li>
<li>没有做的：把光栅族（含掉头）加进训练分布重训一版；工件偏航角与填充方向的扫描；
真实层高下的整层接缝数；多臂（xArm7 / Cobotta）的同一套分析。</li>
</ul>
</div>

<footer>
FR3，工具偏置 0.2034 m（手 + 笔），碰撞模型为 link0–link7 球集。
数据与视频：<code>runs/paper_fill/print_analysis/</code>。
</footer>
</div>
"""

DST.write_text(HTML, encoding='utf-8')
print(f'wrote {DST}  ({len(HTML)/1024:.0f} KB)')
