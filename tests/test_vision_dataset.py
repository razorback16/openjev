"""Source licensing, visual input isolation, and scene-label invariants."""
import base64
import hashlib
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip('PIL')
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_vision_pool as b
import prepare_vision_photos as photos
from vision_request import request_for


def test_license_must_belong_to_this_photo_and_be_permissive():
    def page(photo,license):
        return '<script type="application/ld+json">'+json.dumps({'@graph':[{'@type':'ImageObject','acquireLicensePage':f'https://www.flickr.com/photos/user/{photo}', 'license':license}]})+'</script>'
    landing='https://www.flickr.com/photos/user/123/'
    obj,license=photos.image_object(page('123','https://creativecommons.org/licenses/by/2.0/'),landing)
    assert obj and license=='CC-BY-2.0'
    for url in ['https://creativecommons.org/licenses/by-nc/4.0/','https://creativecommons.org/licenses/by-sa/4.0/']:
        assert photos.image_object(page('123',url),landing)==(None,None)
    assert photos.image_object(page('456','https://creativecommons.org/licenses/by/2.0/'),landing)==(None,None)


def test_clevr_functional_program_checks_relations():
    scene={'objects':[{'color':'red','shape':'cube'},{'color':'blue','shape':'sphere'}],
           'relationships':{'left':[[1],[]]}}
    program=[{'function':'scene','inputs':[]},
             {'function':'filter_color','inputs':[0],'value_inputs':['red']},
             {'function':'unique','inputs':[1]},
             {'function':'relate','inputs':[2],'value_inputs':['left']},
             {'function':'count','inputs':[3]}]
    assert b.clevr_execute(program,scene)=='1'
    program[-1]={'function':'exist','inputs':[3]}
    assert b.clevr_execute(program,scene)=='yes'


def test_chess_text_does_not_reveal_fen_or_mate_answer(monkeypatch):
    monkeypatch.setattr(b,'save_image',lambda row,image:row)
    parent={'id':'source','group_id':'game','split':'validation','vision_split':'evaluation',
            'state':'FEN secret','questions':{'q1':{'type':'choice','instructions':'Move','criteria':{'Qh7#':'mate','Qg7+':'check'}}},
            'targets':{'q1':'Qh7#'},'provenance':{'task':'chess','license':'CC0-1.0','fen':'7k/8/5KQ1/8/8/8/8/8 w - -'}}
    row=b.render_game(parent)
    assert '7k/' not in row['state']
    assert all('#' not in v and '+' not in v for v in row['questions']['q1']['criteria'].values())
    assert row['group_id']=='game' and row['split']=='evaluation'
    assert row['provenance']['move_mapping'][row['targets']['q1']]=='Qh7#'


def test_request_contains_pixels_but_no_supervision(tmp_path):
    path=tmp_path/'picture.png';Image.new('RGB',(128,128),'red').save(path)
    row={'id':'a','state':'Read the picture','questions':{'q1':{'type':'noul','instructions':'Is it red?'}},
         'targets':{'q1':'yes'},'provenance':{'answer':'secret'},
         'images':[{'path':'picture.png','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width':128,'height':128,'content_type':'image/png'}]}
    normal=request_for(row,tmp_path);blank=request_for(row,tmp_path,'blank')
    assert set(normal)=={'model','state','questions','images'}
    assert base64.b64decode(normal['images'][0]['base64'])==path.read_bytes()
    assert blank['images'][0]['base64']!=normal['images'][0]['base64']
    assert blank['state']==normal['state'] and blank['questions']==normal['questions']
    from openjev.api import SystemOneRequest,image_parts
    from openjev.config import Settings
    request=SystemOneRequest.model_validate(normal)
    assert len(image_parts(request.images,Settings()))==1
    with pytest.raises(ValueError):request_for(row,tmp_path,'mismatch',row)


@pytest.mark.parametrize('kind_index',[0,1,2,3,4])
def test_structured_answers_survive_layout_change(monkeypatch,kind_index):
    monkeypatch.setattr(b,'save_image',lambda row,image:row)
    train=b.render_structured((kind_index,'train'));evaluation=b.render_structured((kind_index,'evaluation'))
    # Labels derive from the same data; an alternate rendering cannot change the answers.
    assert train['targets']==evaluation['targets']
    assert train['questions']==evaluation['questions']
