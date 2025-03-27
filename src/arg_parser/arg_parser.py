import os
import argparse
import pathlib

def get_STATS_arg_parser():
    parser = argparse.ArgumentParser(description='ALECE')
    parser.add_argument('--data', type=str, default='STATS',
                        help='Dataset identifier (STATS, STATS, etc.)')
    # ----------------------------------- Data Path Params -----------------------------------
    parser.add_argument('--base_dir', type=str, default='../data/STATS/',
                        help='Base directory for data')
    parser.add_argument('--absolute_base_dir', type=str,
                        default='$WORKSPACE_DIR$/data/STATS',
                        help='Absolute base directory')
    parser.add_argument('--data_dir', type=str, default='../data/STATS/data',
                        help='Data directory')
    parser.add_argument('--workload_base_dir', type=str, default='../data/STATS/workload/',
                        help='Workload base directory')
    parser.add_argument('--source_ckpt_dir', type=str, default=None,
                        help='Path to source checkpoint directory for transfer learning.')
    parser.add_argument('--data_dirname', type=str, default='data',
                        help='Data directory name')
    parser.add_argument('--int_data_dirname', type=str, default='int',
                        help='Intermediate data directory name')
    parser.add_argument('--experiments_dir', type=str, default='../exp/STATS/',
                        help='Directory for experimental results')
    parser.add_argument('--feature_data_dirname', type=str, default='features',
                        help='Feature data directory name')
    parser.add_argument('--workload_fname', type=str, default='workload.sql',
                        help='Workload filename')
    parser.add_argument('--train_queries_fname', type=str, default='train_queries.sql',
                        help='Train queries filename')
    parser.add_argument('--train_sub_queries_fname', type=str, default='train_sub_queries.sql',
                        help='Train sub-queries filename')
    parser.add_argument('--train_single_tbls_fname', type=str, default='train_single_tbls.sql',
                        help='Train single tables filename')
    parser.add_argument('--test_queries_fname', type=str, default='test_queries.sql',
                        help='Test queries filename')
    parser.add_argument('--test_sub_queries_fname', type=str, default='test_sub_queries.sql',
                        help='Test sub-queries filename')
    parser.add_argument('--test_single_tbls_fname', type=str, default='test_single_tbls.sql',
                        help='Test single tables filename')
    parser.add_argument('--base_queries_fname', type=str, default='base_queries.sql',
                        help='Base queries filename')
    parser.add_argument('--tables_info_fname', type=str, default='tables_info.txt',
                        help='Tables info filename')
    # ----------------------------------- DB Params -----------------------------------
    parser.add_argument('--db_data_dir', type=str, default='$PG_DATADIR$', help='Database data directory')
    parser.add_argument('--db_name', type=str, default='', help='Database name')
    parser.add_argument('--db_subqueries_fname', type=str, default='join_sub_queries.txt', help='DB subqueries filename')
    parser.add_argument('--db_single_tbls_fname', type=str, default='single_sub_queries.txt', help='DB single tables filename')
    # ----------------------------------- Model Params -----------------------------------
    parser.add_argument('--model', type=str, default='ALECE', help='Model identifier')
    parser.add_argument('--input_dim', type=int, default=97, help='Input dimension')
    parser.add_argument('--use_float64', type=int, default=0, help='Use float64 precision (1=yes, 0=no)')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--mlp_num_layers', type=int, default=6, help='Number of MLP layers')
    parser.add_argument('--mlp_hidden_dim', type=int, default=512, help='Number of neurons per MLP layer')
    parser.add_argument('--use_positional_embedding', type=int, default=0, help='Use positional embedding (1=yes, 0=no)')
    parser.add_argument('--use_dropout', type=int, default=0, help='Use dropout (1=yes, 0=no)')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--num_attn_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--attn_head_key_dim', type=int, default=511, help='Key dimension for attention heads')
    parser.add_argument('--feed_forward_dim', type=int, default=2048, help='Feed-forward network dimension')
    parser.add_argument('--num_self_attn_layers', type=int, default=6, help='Number of self-attention layers')
    parser.add_argument('--num_cross_attn_layers', type=int, default=6, help='Number of cross-attention layers')
    # ----------------------------------- Featurization Params -----------------------------------
    parser.add_argument('--num_tables', type=int, default=8, help='Number of tables')
    parser.add_argument('--n_bins', type=int, default=40, help='Number of bins for histogram features')
    parser.add_argument('--histogram_feature_dim', type=int, default=430, help='Histogram feature dimension')
    parser.add_argument('--num_attrs', type=int, default=43, help='Number of attributes')
    parser.add_argument('--query_part_feature_dim', type=int, default=96, help='Query part feature dimension')
    parser.add_argument('--join_pattern_dim', type=int, default=11, help='Join pattern dimension')
    # ----------------------------------- Training Params -----------------------------------
    parser.add_argument('--gpu', type=int, default=1, help='GPU id (if available)')
    parser.add_argument('--debug_mode', type=int, default=0, help='Set to 1 for debug mode (short test)')
    parser.add_argument('--buffer_size', type=int, default=32, help='Buffer size for data loading')
    parser.add_argument('--use_loss_weights', type=int, default=1, help='Use loss weights (1=yes, 0=no)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--shuffle_buffer_size', type=int, default=400, help='Shuffle buffer size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--min_n_epochs', type=int, default=3, help='Minimum number of epochs')
    parser.add_argument('--card_log_scale', type=int, default=1, help='Apply logarithmic scaling to cardinalities (1=yes)')
    parser.add_argument('--scaling_ratio', type=float, default=20., help='Scaling ratio for log(card)')
    parser.add_argument('--pretrained_ckpt_dir', type=str, default=None, help='Path to pretrained checkpoint directory')
    parser.add_argument('--freeze_encoder', type=int, default=0, help='Freeze encoder during fine-tuning (1=yes)')
    parser.add_argument('--target_data', type=str, default=None, help='Target dataset for fine-tuning')
    parser.add_argument('--source_dataset', type=str, default=None, help='Source dataset name for transfer learning')
    parser.add_argument('--target_dataset', type=str, default=None, help='Target dataset name for transfer learning')
    # ----------------------------------- Workload Params -----------------------------------
    parser.add_argument('--wl_data_type', type=str, default='init', help='Workload data type (train or test)')
    parser.add_argument('--wl_type', type=str, default='ins_heavy', help='Workload type (ins_heavy, upd_heavy, dist_shift)')
    parser.add_argument('--test_wl_type', type=str, default=None, help='Test workload type')
    # ----------------------------------- e2e Params -----------------------------------
    parser.add_argument('--db_task', type=str, default='query_exec', help='Database task')
    parser.add_argument('--e2e_dirname', type=str, default='e2e', help='End-to-end results directory name')
    parser.add_argument('--e2e_print_sub_queries', type=int, default=0, help='Print sub queries (1=yes)')
    parser.add_argument('--e2e_write_pg_join_cards', type=int, default=0, help='Write PG join cardinalities (1=yes)')
    parser.add_argument('--ignore_single_cards', type=int, default=1, help='Ignore single table cardinalities (1=yes)')
    # ----------------------------------- Checkpoint Params -----------------------------------
    parser.add_argument('--ckpt_dirname', type=str, default='ckpt/{0:s}_{1:s}', help='Checkpoint directory name pattern')
    parser.add_argument('--keep_train', type=int, default=0, help='Keep training (1=yes)')
    # ----------------------------------- P-error Params -----------------------------------
    parser.add_argument('--costs_dirname', type=str, default='costs', help='Costs directory name')
    parser.add_argument('--hints_dirname', type=str, default='hints', help='Hints directory name')
    # ----------------------------------- Calc Params -----------------------------------
    parser.add_argument('--calc_task', type=str, default='q_error', help='Calculation task (e.g., q_error)')

    args = parser.parse_args()
    return args

def get_arg_parser():
    args = get_STATS_arg_parser()
    workspace_dir = str(pathlib.Path().resolve().parent.absolute())
    args.absolute_base_dir = args.absolute_base_dir.replace('$WORKSPACE_DIR$', workspace_dir)
    if args.test_wl_type is None:
        args.test_wl_type = args.wl_type
    if args.test_wl_type == 'static':
        args.db_name = args.data.lower()
    else:
        terms = args.wl_type.split('_')
        wl_type_pre = terms[0]
        terms = args.test_wl_type.split('_')
        test_wl_type_pre = terms[0]
        args.db_name = f'{args.data}_{args.model}_{wl_type_pre}_{test_wl_type_pre}'.lower()
    return args

if __name__ == '__main__':
    args = get_arg_parser()
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
