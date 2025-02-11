from alphapose_module.alphapose.utils.metrics import evaluate_mAP
import sys

sysout = sys.stdout

res = evaluate_mAP('pretrain_model/validate_kpt_epoch0433.json', ann_type='keypoints', ann_file='data/person_visible_keypoints_val2017.json', halpe=None)

sys.stdout = sysout

print(res)