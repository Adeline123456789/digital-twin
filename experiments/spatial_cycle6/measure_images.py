from pathlib import Path
import json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug/self_evolution/spatial_cycle6');cap=B/'real_20261116T062116Z';out=B/'image_disparity';out.mkdir(exist_ok=False)

def components(mask,area):
 n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype('uint8'))
 return [{'area_px':int(stats[i,4]),'center_px':centers[i].tolist(),'bounds_xywh':stats[i,:4].tolist()} for i in range(1,n) if area[0]<=stats[i,4]<=area[1]]

def measure(p,region):
 im=cv2.imread(str(p));g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);xy=np.indices(g.shape);result={}
 if region=='holder':
  roi=(xy[1]>=180)&(xy[1]<340)&(xy[0]<150)
  for v in [30,40,50]:
   c=components((g<v)&roi,(500,2500));result[f'black_rectangle_threshold{v}']=c
  yellow=cv2.inRange(hsv,np.array([10,80,60]),np.array([40,255,255]));c,_=cv2.findContours(yellow,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);hull=np.zeros_like(g);cv2.drawContours(hull,[max(c,key=cv2.contourArea)],-1,255,-1)
  for v in [30,40,50]:
   c=components((g<v)&(hull>0)&(xy[0]>200),(150,2000));result[f'front_hole_threshold{v}']=c
 else:
  blue=cv2.inRange(hsv,np.array([90,50,30]),np.array([125,255,255]));c=components(blue>0,(5000,30000));result['blue_panels']=[x for x in c if 150<x['center_px'][0]<370 and x['center_px'][1]<330]
 return result
result={}
for region,frame in [('holder',561),('socket',1496)]:
 paths={'real':cap/'right_wrist'/f'{frame:04d}.jpg','simulation':B/'probe_002'/f'{region}_center_hold_right_wrist.png'}
 result[region]={name:measure(p,region) for name,p in paths.items()};result[region]['image_sha256']={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()}
 canvas=Image.new('RGB',(1280,520),'white');draw=ImageDraw.Draw(canvas)
 for col,(name,p) in enumerate(paths.items()):
  canvas.paste(Image.open(p),(640*col,40));draw.text((640*col+12,10),f'{region}: {name} (same frozen probe hold)',fill='black')
 canvas.save(out/(region+'_comparison.png'))
(out/'measurement_candidates.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
