import wandb
import numpy as np
import argparse
import json

def process_tta_res(original_pred_file, tta_files, with_wandb=True):    
    original_preds = json.load(open(original_pred_file, 'r'))
    all_instances_keys = original_preds.keys()
    tta_correct = {}
    print("Computing TTA accuracy")
    for tta_file in tta_files:
        tta_preds = json.load(open(tta_file, 'r'))
        print(f"Processing TTA file {tta_file}")
        for instance, pred in tta_preds.items():
            assert instance not in tta_correct, f"Repeated instance {instance} across TTA files"
            if 'sounds' in original_pred_file:
                tta_correct[instance] = pred['pred'] == pred['label']
            elif 'EPIC' or 'STA' in original_pred_file:
                tta_correct[instance] = {}
                tta_correct[instance]['noun'] = pred['noun']['pred'] == pred['noun']['label']
                if not "STA" in original_pred_file:
                    tta_correct[instance]['verb'] = pred['verb']['pred'] == pred['verb']['label'] 
    print(len(tta_correct))
    print(len(all_instances_keys))
    # assert len(tta_correct) == len(all_instances_keys), f"Some instances are missing in TTA files: {set(all_instances_keys) - set(tta_correct.keys())}"
    print(f"TTA accuracy computed for {len(tta_correct)} instances")
    
    if 'sounds' in original_pred_file:
        accuracy = np.mean(list(tta_correct.values()))
    elif 'EPIC'  or 'STA' in original_pred_file:
        # Repport to wandb verb and noun accuracy
        accuracy_noun = np.mean([v['noun'] for v in tta_correct.values()])
        accuracy_verb = 0
        if not "STA" in original_pred_file:
            accuracy_verb = np.mean([v['verb'] for v in tta_correct.values()])
        if with_wandb:
            wandb.log({'tta_noun_accuracy': accuracy_noun, 'tta_verb_accuracy': accuracy_verb})
        print(accuracy_noun)
        # Report accuracy as verb accuracy for EPIC
        if 'EPIC' in original_pred_file:
            accuracy = accuracy_verb
        else:
            accuracy = accuracy_noun
    return accuracy
    


def init_wandb(args):
    if args.with_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        wandb.config.update(args)
    
def main():
    parser = argparse.ArgumentParser(description='Process TTA results')
    parser.add_argument('--original_pred_file', type=str, help='Path to the original prediction file')
    parser.add_argument('--tta_files', type=str, nargs='+', help='Paths to the TTA prediction files')
    parser.add_argument('--with_wandb', default=False, type=bool, help='Using Wandb for logging')
    parser.add_argument('--wandb_project', default='tta-mm', type=str, help='Wandb project name')
    parser.add_argument('--wandb_entity', default='username',type=str, help='Wandb entity name')
    parser.add_argument('--config-lambda-mi', type=float, help='Lambda value for MI')
    parser.add_argument('--config-lambda-kl', type=float, help='Lambda value for KL')
    parser.add_argument('--config-tta-lr', type=float, help='Base learning rate')
    parser.add_argument('--seed', type=int, help='seed for reproducibility')
    args = parser.parse_args()
    
    init_wandb(args)
    
    accuracy = process_tta_res(args.original_pred_file, args.tta_files, args.with_wandb)
    if args.with_wandb:
        wandb.log({'tta_accuracy': accuracy, 'seed': args.seed})
    # Log accuracy as a number in wandb
    print('\n-----------------------------------')
    print(f'TTA accuracy: {accuracy*100:.2f}%')
    print('-----------------------------------\n')
if __name__ == '__main__':
    main()