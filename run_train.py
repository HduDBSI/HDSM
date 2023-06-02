# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse  #它允许您定义从命令行运行Python脚本时可以传递给脚本的参数

from recbole.quick_start import run_recbole #run_recbole函数是RecBole为快速运行不同配置设置的实验提供的方便函数。

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='LightSANs', help='name of models') #add_argument方法定义脚本接受的每个参数
    parser.add_argument('--dataset', '-d', type=str, default='xiami', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')
    parser.add_argument('--method', type=str, default='None', \
                        help='None, CL4SRec, CL4SRec_XAUG, DuoRec, DuoRec_XAUG, ...')
    parser.add_argument('--cl_loss_weight', type=float, default=0.1, help='weight for contrastive loss')
    parser.add_argument('--temp_ratio', type=float, default=1.0, help='temperature ratio')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

    ### ours
    #saliency 方法用于计算每个输入特征对输出结果的贡献程度，从而确定哪些特征最重要。occlusion
    #方法则是将输入特征部分遮挡，然后观察输出结果的变化，以此来确定哪些特征对输出结果最为敏感。
    parser.add_argument('--xai_method', type=str, default='occlusion', help='saliency, occlusion')
    parser.add_argument('--seq_long', type=int, default=25)
    parser.add_argument('--seq_short', type=int, default=5)
    parser.add_argument('--fusion_type', type=int, default=1)
    parser.add_argument('--ablation', type=str, default='all')

    args, _ = parser.parse_known_args()

    if args.dataset == 'ml-1m':
        args.fusion_type = 1
        args.seq_long = 15
    if args.dataset == 'ml-1m-2':
        args.fusion_type = 1
        args.seq_long = 15
    if args.dataset == 'xiami':
        args.fusion_type = 1
        args.seq_long = 15
    if args.dataset == '30music':
        args.seq_long = 25

    config_dict = {
        'neg_sampling': None,  #bpr_loss要注释这一行
        'method': args.method,
        'cl_loss_weight': args.cl_loss_weight,
        'temp_ratio': args.temp_ratio,
        'gpu_id': args.gpu_id,
        'seq_long': args.seq_long,
        'seq_short': args.seq_short,
        'fusion_type': args.fusion_type,
        'ablation': args.ablation,
        'xai_method': args.xai_method,
    }

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None #包含模型和训练参数的配置文件列表
    run_recbole(model=args.model, dataset=args.dataset, method=args.method,
                config_file_list=config_file_list, config_dict=config_dict)
    #模型根据类名来调用
