"""Build the 100K-image decision pool; see dataset/vision/METHODOLOGY.md."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import copy
import functools
import gzip
import hashlib
import io
import json
import random
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/openjev-vision'
SOURCE, RELEASE = BASE/'source', BASE/'release'
SEED = 20260920
FONT_DIR = ROOT/'dataset/vision/assets'
SPLITS = ('train','calibration','evaluation')


def digest(*args):
    return hashlib.sha256(json.dumps(args,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@functools.lru_cache(maxsize=64)
def font(size, bold=False, serif=False):
    name = 'DejaVuSerif.ttf' if serif else 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(str(FONT_DIR/name),size)


def text(draw, xy, value, size=22, color='#17202a', bold=False, anchor=None, serif=False):
    draw.text(xy,str(value),font=font(size,bold,serif),fill=color,anchor=anchor)


def palette(split, rng):
    # The evaluation set uses a separate color/font theme, never a training variant.
    colors=rng.choice([('#f5f7fa','#d4e3ee','#447d9b','#12344a'),('#fffaf0','#eadac5','#a27743','#503924'),
                       ('#f2faf5','#cae3d5','#4d8c6c','#244f39')])
    return ('#f4f0fa','#e0d5ed','#795a9b','#422d58') if split=='evaluation' else colors


def canvas(split, rng, title, size=(960,720)):
    colors = palette(split,rng)
    image = Image.new('RGB',size,colors[0]); draw = ImageDraw.Draw(image)
    text(draw,(35,24),title,28,bold=True,serif=split=='evaluation')
    return image,draw,colors


def row_base(family, key, split, state, questions, targets, provenance, group=None):
    row = {'id':digest('vision-v1',family,key),'group_id':group or digest(family,key),'split':split,
           'state':state,'questions':questions,'targets':targets,
           'provenance':{'family':family,'license':'Apache-2.0',**provenance}}
    return row


def choice(instruction, choices, rng):
    items=list(choices.items());rng.shuffle(items)
    return {'type':'choice','instructions':instruction,'criteria':dict(items)}


def save_image(row,image):
    relative=Path('images')/row['provenance']['family']/f'{row["id"]}.png'
    path=RELEASE/relative;path.parent.mkdir(parents=True,exist_ok=True)
    image.save(path,compress_level=3)
    row['images']=[{'path':str(relative),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'width':image.width,'height':image.height,'content_type':'image/png'}]
    return row


def chess_board(draw,fen,colors,reverse=False):
    cells=[]
    for row in fen.split()[0].split('/'):
        parsed=[]
        for ch in row:
            parsed.extend(['.']*int(ch) if ch.isdigit() else [ch])
        assert len(parsed)==8;cells.extend(parsed)
    x0,y0,side=200,98,66
    symbols={'k':'♚','q':'♛','r':'♜','b':'♝','n':'♞','p':'♟'}
    for i,piece in enumerate(cells):
        r,c=divmod(i,8);rr,cc=(7-r,7-c) if reverse else (r,c)
        x,y=x0+cc*side,y0+rr*side
        draw.rectangle((x,y,x+side,y+side),fill=colors[1] if (r+c)%2==0 else colors[2])
        if piece!='.':
            white=piece.isupper()
            draw.text((x+side/2,y+side/2),symbols[piece.lower()],font=font(49),anchor='mm',
                      fill='#ffffff' if white else '#111111',stroke_width=1,stroke_fill='#111111')
    for j in range(8):
        text(draw,(x0+j*side+side/2,y0+8*side+22),chr(97+(7-j if reverse else j)),20,anchor='mm')
        text(draw,(x0-18,y0+j*side+side/2),j+1 if reverse else 8-j,20,anchor='mm')
    return cells


def render_game(parent):
    source=parent['provenance'];game=source['task'];split=parent['vision_split']
    rng=random.Random(digest(SEED,parent['id']))
    image,draw,colors=canvas(split,rng,{'chess':'Chess','gomoku':'Freestyle Gomoku','connect_four':'Connect Four','texas_holdem':"Texas Hold'em"}[game])
    questions=copy.deepcopy(parent['questions']);targets=copy.deepcopy(parent['targets'])
    p={'parent_record_id':parent['id'],'parent_group_id':parent['group_id'],'parent_split':parent['split'],
       'game':game,'source_license':source['license'],'render_theme':'heldout-purple-serif' if split=='evaluation' else 'training-palette',
       'render_data':source}
    if game=='chess':
        fen=source['fen'];parts=fen.split();reverse=bool(rng.randrange(2))
        chess_board(draw,fen,colors,reverse)
        rights=parts[2];ep=parts[3]
        text(draw,(200,680),f'{"White" if parts[1]=="w" else "Black"} to move   Castling: {rights}   En passant: {ep}',17)
        state=f'Use the chessboard image. {"White" if parts[1]=="w" else "Black"} to move. '
        state+=f'Castling rights: {rights}. En-passant target: {ep}. Move history is unavailable. '
        state+='Standard chess. White pieces have white fill and a black outline; Black pieces have black fill.'
        original=list(questions['q1']['criteria'])
        mapping={move:f'move_{i+1}' for i,move in enumerate(original)}
        descriptions={mapping[m]:re.sub(r'[+#]+$','',m) for m in original}
        questions={'q1':choice('Choose the strongest legal move for the side to move. Moves use SAN with check/mate suffixes omitted.',descriptions,rng)}
        targets={'q1':mapping[parent['targets']['q1']]}
        p['move_mapping']={v:k for k,v in mapping.items()};p['orientation']='black_bottom' if reverse else 'white_bottom'
        p['acceptable_targets']=[mapping[x] for x in parent.get('acceptable_targets',{}).get('q1',[parent['targets']['q1']])]
    elif game=='gomoku':
        board=source['board'];x0,y0,step=180,105,36
        draw.rectangle((x0-15,y0-15,x0+14*step+15,y0+14*step+15),fill=colors[1])
        for i in range(15):
            draw.line((x0+i*step,y0,x0+i*step,y0+14*step),fill=colors[3],width=1)
            draw.line((x0,y0+i*step,x0+14*step,y0+i*step),fill=colors[3],width=1)
            text(draw,(x0+i*step,y0+14*step+29),chr(65+i),16,anchor='mm')
            text(draw,(x0-28,y0+i*step),15-i,16,anchor='mm')
        for i,v in enumerate(board):
            if not v:continue
            r,c=divmod(i,15);x,y=x0+c*step,y0+r*step
            draw.ellipse((x-13,y-13,x+13,y+13),fill='#171717' if v==1 else '#ffffff',outline='#333333',width=2)
        text(draw,(180,668),'X = black stones; O = white stones',19)
        state=f'Use the board image. Freestyle 15x15 Gomoku: five or more in a row wins, no forbidden moves. '
        state+=f'X is black and plays first; O is white. {"X" if source["side_to_move"]==1 else "O"} to move. A15 is top-left.'
    elif game=='connect_four':
        a,b=source['current_bits'],source['other_bits'];current=source['side_to_move']
        x0,y0,step=185,112,78
        draw.rounded_rectangle((x0-12,y0-12,x0+7*step+12,y0+6*step+12),radius=12,fill=colors[2])
        for r in range(6):
            for c in range(7):
                bit=1<<(c*7+5-r)
                piece=current if a&bit else ('O' if current=='X' else 'X') if b&bit else '.'
                x,y=x0+c*step+step/2,y0+r*step+step/2
                fill='#d8473f' if piece=='X' else '#f6ce48' if piece=='O' else colors[0]
                draw.ellipse((x-30,y-30,x+30,y+30),fill=fill,outline=colors[3],width=2)
        for c in range(7):text(draw,(x0+c*step+step/2,y0+6*step+30),c+1,22,anchor='mm')
        state=f'Use the Connect Four board. Red = X, yellow = O. X plays first; {current} to move. Four in a row wins. Columns 1–7; gravity applies.'
    else:
        board,hero=source['board'],source['hero']
        suits={'c':'♣','d':'♦','h':'♥','s':'♠'}
        def card(c,x,y):
            draw.rounded_rectangle((x,y,x+105,y+155),radius=12,fill='white',outline=colors[3],width=3)
            color='#b52a38' if c[1] in 'dh' else '#17202a'
            text(draw,(x+15,y+10),c[0],32,color,bold=True)
            text(draw,(x+53,y+91),suits[c[1]],48,color,anchor='mm')
        text(draw,(140,93),'Community board',24,bold=True)
        for i,c in enumerate(board):card(c,140+i*130,140)
        text(draw,(330,350),'Your hole cards',24,bold=True)
        for i,c in enumerate(hero):card(c,330+i*150,398)
        state=re.sub(r'Your hole cards: .*?\. Board: .*?\.', 'Your hole cards and community board are shown in the image.',parent['state'],count=1)
        assert 'Your hole cards:' not in state
    row=row_base('games',parent['id'],split,state,questions,targets,p,parent['group_id'])
    return save_image(row,image)


def select_games():
    frozen=SOURCE/'games.selected.jsonl.gz'
    if frozen.exists():
        with gzip.open(frozen,'rt') as f:return [json.loads(line) for line in f]
    quotas={'chess':{'train':10500,'validation':750,'calibration':750},
            'gomoku':{'train':5700,'validation':650,'calibration':650},
            'connect_four':{'train':2500,'validation':251,'calibration':249},
            'texas_holdem':{'train':2200,'validation':400,'calibration':400}}
    pools=collections.defaultdict(list)
    for base in ('openjev-games','openjev-holdem'):
        for split in ('train','validation','calibration'):
            for line in (ROOT/'data'/base/'release'/f'{split}.jsonl').open():
                row=json.loads(line);game=row['provenance']['task']
                if game not in quotas or 'target_distributions' in row:continue
                pools[(game,split)].append(row)
    selected=[]
    for game,counts in quotas.items():
        for split,n in counts.items():
            candidates=sorted(pools[(game,split)],key=lambda r:digest('vision-select',r['id']))
            assert len(candidates)>=n,(game,split,len(candidates),n)
            for row in candidates[:n]:
                row['vision_split']='evaluation' if split=='validation' else split
                selected.append(row)
    assert len(selected)==25000
    return selected


def render_structured(spec):
    index,split=spec;rng=random.Random(digest(SEED,'structured',index));kind=['bar','line','table','invoice','ui'][index%5]
    image,draw,colors=canvas(split,rng,{'bar':'Service requests','line':'Daily throughput','table':'Regional inventory','invoice':'INVOICE','ui':'Deployment console'}[kind])
    p={'generator':'structured-v1','index':index,'kind':kind,'render_theme':'heldout-purple-serif' if split=='evaluation' else 'training-palette'}
    if kind in ('bar','line'):
        labels=rng.sample(['Alpha','Beta','Gamma','Delta','Epsilon','Zeta','Eta','Theta'],6)
        values=rng.sample(range(10,96),6);p.update(labels=labels,values=values)
        x0,y0,width,height=110,585,720,440
        horizontal=kind=='bar' and split=='evaluation'
        for v in range(0,101,20):
            if horizontal:continue
            y=y0-v/100*height;draw.line((x0,y,x0+width,y),fill='#c7cbd1');text(draw,(x0-18,y),v,18,anchor='rm')
        if not horizontal:draw.line((x0,y0,x0+width,y0),fill=colors[3],width=2)
        points=[]
        for j,(label,value) in enumerate(zip(labels,values)):
            x=x0+65+j*115;y=y0-value/100*height
            if horizontal:
                yy=170+j*70;xx=210+value*6
                draw.rectangle((210,yy,xx,yy+35),fill=colors[2],outline=colors[3],width=2)
                text(draw,(185,yy+18),label,19,anchor='rm');text(draw,(xx+12,yy+18),value,20,bold=True,anchor='lm')
                continue
            if kind=='bar':draw.rectangle((x-32,y,x+32,y0),fill=colors[2],outline=colors[3],width=2)
            else:points.append((x,y))
            text(draw,(x,y-18),value,20,bold=True,anchor='mm');text(draw,(x,y0+30),label,18,anchor='mm')
        if points:
            draw.line(points,fill=colors[2],width=5)
            for x,y in points:draw.ellipse((x-7,y-7,x+7,y+7),fill=colors[3])
        target=labels[values.index(max(values))]
        q1=choice('Which labeled category has the highest plotted value?',{s:s for s in labels},rng)
        k=rng.randrange(6);threshold=rng.choice([25,50,75]);p.update(probe_index=k,threshold=threshold)
        q2={'type':'noul','instructions':f'Is the value for {labels[k]} strictly greater than {threshold}?'}
        t2='yes' if values[k]>threshold else 'no'
    elif kind=='table':
        labels=rng.sample(['North','South','East','West','Central','Coastal'],4)
        starts=rng.sample(range(20,200),4);changes=rng.sample(range(-15,65),4);ends=[a+b for a,b in zip(starts,changes)]
        p.update(labels=labels,starts=starts,ends=ends)
        columns=[110,400,675]
        headers=['Region','Closing','Opening'] if split=='evaluation' else ['Region','Opening','Closing']
        for x,label in zip(columns,headers):text(draw,(x,125),label,25,bold=True)
        for j,label in enumerate(labels):
            y=205+j*83;draw.rectangle((80,y-14,870,y+51),fill=colors[1] if j%2==0 else '#ffffff')
            values=[label,ends[j],starts[j]] if split=='evaluation' else [label,starts[j],ends[j]]
            for x,value in zip(columns,values):text(draw,(x,y),value,27)
        target=labels[changes.index(max(changes))]
        q1=choice('Which region has the largest increase from opening to closing inventory?',{s:s for s in labels},rng)
        k=rng.randrange(4);p['probe_index']=k
        q2={'type':'noul','instructions':f'Did {labels[k]} finish with more inventory than it started with?'}
        t2='yes' if ends[k]>starts[k] else 'no'
    elif kind=='invoice':
        qty=[rng.randrange(1,8) for _ in range(4)];price=[rng.randrange(500,6000) for _ in range(4)]
        amounts=[a*b for a,b in zip(qty,price)];subtotal=sum(amounts);tax=subtotal//10;total=subtotal+tax
        p.update(qty=qty,unit_cents=price,subtotal_cents=subtotal,tax_cents=tax,total_cents=total)
        money=lambda n:f'${n/100:,.2f}'
        text(draw,(45,80),f'Invoice INV-{index:06d}',20)
        cols=[60,405,570,780]
        for x,s in zip(cols,['Description','Qty','Unit price','Amount']):text(draw,(x,140),s,20,bold=True)
        for j in range(4):
            y=205+j*67;draw.line((45,y+45,910,y+45),fill=colors[1],width=2)
            for x,s in zip(cols,[f'Item {j+1}',qty[j],money(price[j]),money(amounts[j])]):text(draw,(x,y),s,21)
        for y,label,amount in [(500,'Subtotal',subtotal),(550,'Tax',tax),(615,'Amount due',total)]:
            text(draw,(505,y),label,24,bold=True);text(draw,(900,y),money(amount),24,bold=True,anchor='ra')
        choices=sorted({total,subtotal,total+1234,max(0,total-2345)})
        target=str(total);q1=choice('What is the amount due shown on this invoice?',{str(n):money(n) for n in choices},rng)
        threshold=rng.choice([30000,60000,90000]);p['threshold_cents']=threshold
        q2={'type':'noul','instructions':f'Is the amount due strictly greater than {money(threshold)}?'}
        t2='yes' if total>threshold else 'no'
    else:
        available=['Retry','Cancel','Inspect','Archive','Resume','Pause','Deploy','Rollback','Refresh','Download',
                   'Approve','Reject','Schedule','Duplicate','Export','Open logs','View history','Run checks','Stop job','Edit settings']
        # Injective permutation selection avoids padding the pool with only 24 button orders.
        code=((index//5)*7919+SEED) % (20*19*18*17)
        actions=[]
        for _ in range(4):
            actions.append(available.pop(code % len(available)));code//=len(available)+1
        status=rng.choice(['Failed','Running','Paused','Completed','Queued'])
        target_action=rng.choice(actions);p.update(actions=actions,status=status,target_action=target_action)
        draw.rounded_rectangle((45,105,915,235),radius=12,fill=colors[1]);text(draw,(75,130),f'Job: build-{index:06d}',25,bold=True)
        text(draw,(75,180),f'Status: {status}',25)
        locations=['A','B','C','D']
        for j,action in enumerate(actions):
            x=80+(j%2)*440;y=310+(j//2)*170
            height=110
            if split=='evaluation':x=295;y=265+j*105;height=80
            draw.rounded_rectangle((x,y,x+350,y+height),radius=14,fill=colors[2],outline=colors[3],width=2)
            text(draw,(x+25,y+height/2),locations[j],20,color='white',anchor='mm')
            text(draw,(x+185,y+height/2),action,30,color='white',bold=True,anchor='mm')
        target=locations[actions.index(target_action)]
        q1=choice(f'Which marked button is labeled "{target_action}"?',{k:'Button '+k for k in locations},rng)
        q2={'type':'noul','instructions':'Does the displayed job have status Failed?'};t2='yes' if status=='Failed' else 'no'
    row=row_base('structured',index,split,'Answer using the supplied image.',{'q1':q1,'q2':q2},{'q1':target,'q2':t2},p)
    return save_image(row,image)


def structured_specs():
    order=sorted(range(30000),key=lambda i:digest('structured-split',i))
    assigned={i:'evaluation' if j<1949 else 'calibration' if j<2900 else 'train' for j,i in enumerate(order)}
    return [(i,assigned[i]) for i in range(30000)]


def clevr_execute(program,scene):
    objects=scene['objects'];outputs=[]
    for node in program:
        function=node['function'];args=[outputs[i] for i in node['inputs']];values=node.get('value_inputs',[])
        if function=='scene':value=list(range(len(objects)))
        elif function.startswith('filter_'):
            attr=function[7:];value=[i for i in args[0] if objects[i][attr]==values[0]]
        elif function=='unique':
            assert len(args[0])==1;value=args[0][0]
        elif function=='relate':value=scene['relationships'][values[0]][args[0]]
        elif function.startswith('same_'):
            attr=function[5:];value=[i for i,o in enumerate(objects) if i!=args[0] and o[attr]==objects[args[0]][attr]]
        elif function.startswith('query_'):value=objects[args[0]][function[6:]]
        elif function=='count':value=len(args[0])
        elif function=='exist':value=bool(args[0])
        elif function=='union':value=sorted(set(args[0])|set(args[1]))
        elif function=='intersect':value=sorted(set(args[0])&set(args[1]))
        elif function.startswith('equal_'):value=args[0]==args[1]
        elif function=='less_than':value=args[0]<args[1]
        elif function=='greater_than':value=args[0]>args[1]
        else:raise ValueError(function)
        outputs.append(value)
    value=outputs[-1]
    return ('yes' if value else 'no') if isinstance(value,bool) else str(value)


def build_clevr():
    manifest=json.loads((SOURCE/'clevr-http.json').read_text());members=[x['name'] for x in manifest['members']]
    output=[]
    with zipfile.ZipFile(SOURCE/'CLEVR_selected.sparse.zip') as archive:
        for original_split in ('train','val'):
            scenes={s['image_filename']:s for s in json.loads(archive.read(f'CLEVR_v1.0/scenes/CLEVR_{original_split}_scenes.json'))['scenes']}
            questions=collections.defaultdict(list)
            for q in json.loads(archive.read(f'CLEVR_v1.0/questions/CLEVR_{original_split}_questions.json'))['questions']:
                questions[q['image_filename']].append(q)
            selected=[name for name in members if f'/images/{original_split}/' in name]
            for index,name in enumerate(selected):
                filename=Path(name).name;scene=scenes[filename]
                split='evaluation' if original_split=='val' else 'calibration' if index>=20000 else 'train'
                rng=random.Random(digest('clevr',filename));qs=questions[filename][:];rng.shuffle(qs)
                qs=qs[:3];compiled={};targets={}
                for j,q in enumerate(qs):
                    answer=str(q['answer']);assert clevr_execute(q['program'],scene)==answer
                    key=f'q{j+1}'
                    if answer in ('yes','no'):compiled[key]={'type':'noul','instructions':q['question']}
                    else:
                        op=q['program'][-1]['function']
                        domain={'count':[str(n) for n in range(11)],'query_color':['gray','red','blue','green','brown','purple','cyan','yellow'],
                                'query_shape':['cube','sphere','cylinder'],'query_size':['small','large'],'query_material':['rubber','metal']}[op]
                        compiled[key]=choice(q['question'],{x:x for x in domain},rng)
                    targets[key]=answer
                row=row_base('clevr',filename,split,'Answer using the rendered scene image.',compiled,targets,
                    {'license':'CC-BY-4.0','source':'CLEVR v1.0','source_url':'https://cs.stanford.edu/people/jcjohns/clevr/',
                     'original_split':original_split,'source_image':filename,'source_questions':qs,'scene':scene,
                     'attribution':'Johnson, Hariharan, van der Maaten, Fei-Fei, Zitnick and Girshick; CLEVR, CVPR 2017'})
                data=archive.read(name);image=Image.open(io.BytesIO(data));image.load()
                # Preserve original image bytes; the ZIP reader verifies the source CRC.
                path=Path('images/clevr')/(row['id']+'.png');(RELEASE/path).parent.mkdir(parents=True,exist_ok=True)
                (RELEASE/path).write_bytes(data)
                row['images']=[{'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'width':image.width,'height':image.height,'content_type':'image/png'}]
                output.append(row)
                if len(output)%5000==0:print(f'CLEVR compiled: {len(output)}',flush=True)
    return output


def build_photos():
    output=[]
    for line in (SOURCE/'photos/selection.jsonl').open():
        p=json.loads(line);rng=random.Random(digest('photo',p['image_id']))
        pos=rng.choice(p['labels']['positive']);neg=rng.choice(p['labels']['negative'])
        pairs=[(pos,'yes'),(neg,'no')];rng.shuffle(pairs)
        questions={f'q{i+1}':{'type':'noul','instructions':f'Does the image contain {label}?'} for i,(label,_) in enumerate(pairs)}
        targets={f'q{i+1}':answer for i,(_,answer) in enumerate(pairs)}
        row=row_base('photos',p['image_id'],p['split'],'Answer using the photograph.',questions,targets,
                     {'license':p['license'],'source':'Open Images human-verified image labels','source_image_id':p['image_id'],
                      'original_split':p['original_split'],'license_evidence':p['license_evidence'],'original_metadata':p['source_metadata'],
                      'license_checked_utc':p['license_checked_utc'],'landing_html_sha256':p['landing_html_sha256'],
                      'annotation_license':'CC-BY-4.0','selected_labels':pairs,'all_verified_labels':p['labels'],
                      'modifications':p['modifications'],'download_url':p['download_url'],'download_sha256':p['download_sha256']})
        relative=Path('images/photos')/(row['id']+'.jpg');(RELEASE/relative).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE/'photos/images'/f'{p["image_id"]}.jpg',RELEASE/relative)
        row['images']=[{'path':str(relative),'sha256':p['image_sha256'],'width':p['dimensions'][0],'height':p['dimensions'][1],'content_type':'image/jpeg'}]
        output.append(row)
    assert len(output)==20000
    return output


def write_part(name, rows):
    (RELEASE/'parts').mkdir(parents=True,exist_ok=True)
    (RELEASE/'parts'/f'{name}.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    print(f'{name}: {len(rows)} image records',flush=True)


def configure(source,output):
    global SOURCE,RELEASE
    SOURCE,RELEASE=Path(source),Path(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--part',choices=['games','structured','clevr','photos'],required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output',type=Path,default=RELEASE)
    args=p.parse_args();configure(args.source,args.output);RELEASE.mkdir(parents=True,exist_ok=True)
    if args.part in ('games','structured'):
        function,inputs=(render_game,select_games()) if args.part=='games' else (render_structured,structured_specs())
        rows=[]
        with concurrent.futures.ProcessPoolExecutor(args.workers,initializer=configure,initargs=(SOURCE,RELEASE)) as ex:
            for i,row in enumerate(ex.map(function,inputs,chunksize=32)):
                rows.append(row)
                if (i+1)%5000==0:print(f'{args.part}: rendered {i+1}',flush=True)
    else:rows=build_clevr() if args.part=='clevr' else build_photos()
    write_part(args.part,rows)


if __name__=='__main__':main()
