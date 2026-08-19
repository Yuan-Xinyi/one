source ~/miniconda3/etc/profile.d/conda.sh && conda activate one && python - << 'EOF'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

C='#1f77b4'; A='#d62728'; PL='#9ecae1'
s=np.linspace(0,1,200)
OUT='/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05/Yuan/IJRR/2026_Yuan_RAL/imgs/'

def render(name,xyz,axes_dirs,legend=False,k=6):
    fig=plt.figure(figsize=(3.6,2.9))
    ax=fig.add_subplot(111,projection='3d')
    x,y,z=xyz
    XX,YY=np.meshgrid([-0.08,1.12],[-0.34,0.34])
    ZZ=np.zeros_like(XX)
    ax.plot_surface(XX,YY,ZZ,color=PL,alpha=0.30,linewidth=0,shade=False)
    ax.plot(x,y,z,color=C,lw=2.8,zorder=5)
    idx=np.linspace(8,len(x)-9,k).astype(int)
    for j in idx:
        u,v,w=axes_dirs[:,j]
        ax.quiver(x[j],y[j],z[j],u,v,w,length=0.22,color=A,
                  arrow_length_ratio=0.35,lw=1.7)
    if legend:
        ax.text(0.02,-0.02,0.40,'cone axis $\\mathbf{n}(s)$',color=A,fontsize=10)
        ax.text(0.78,-0.55,0.0,'task plane',color='#4a7fb5',fontsize=10)
    ax.set_xlim(-0.05,1.1); ax.set_ylim(-0.4,0.4); ax.set_zlim(-0.02,0.42)
    ax.set_box_aspect((1.35,1.0,0.55))
    ax.view_init(elev=24,azim=-63)
    ax.set_axis_off()
    fig.subplots_adjust(left=-0.18,right=1.18,top=1.25,bottom=-0.25)
    fig.savefig(OUT+name,dpi=300,facecolor='white',pil_kwargs={'quality':95})
    plt.close(fig)
    print('wrote',name)

zz=np.zeros_like(s); oo=np.ones_like(s)
nrm=np.stack([zz,zz,oo])
render('path_straight.jpg',(s,zz,zz),nrm,legend=True)

th=s*np.pi*0.5
x=1.05*np.sin(th); y=0.62*(1-np.cos(th))-0.28; z=zz
render('path_arc.jpg',(x,y,z),nrm)

x=s; y=0.18*np.sin(2*np.pi*2.2*s)
render('path_serpentine.jpg',(x,y,zz),nrm)

ang=s*np.pi*0.42
nr=np.stack([zz, np.sin(ang), np.cos(ang)])
render('path_rotating.jpg',(s,zz,zz),nr)

from PIL import Image
ims=[Image.open(OUT+f'path_{n}.jpg') for n in ['straight','arc','serpentine','rotating']]
w=sum(i.width for i in ims); h=max(i.height for i in ims)
cat=Image.new('RGB',(w,h),'white'); xx=0
for i in ims: cat.paste(i,(xx,0)); xx+=i.width
cat.save('/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/preview_row.jpg',quality=90)
print('preview updated')
EOF