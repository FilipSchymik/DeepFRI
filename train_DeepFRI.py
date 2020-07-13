import pickle
import argparse
import glob
import numpy as np

from deepfrier.utils import load_catalogue, rnd_adj, norm_adj

from deepfrier.DeepFRI import DeepFRI
from deepfrier.utils import seq2onehot


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden_dims', type=int, default=[], nargs='+', help="Dimensions of hidden layers.")
    parser.add_argument('--gcn_dims', type=int, default=[128, 128, 256], nargs='+', help="Dimensions of GCN layers.")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate.")
    parser.add_argument('--l2_reg', type=float, default=1e-3, help="L2 regularization coefficient.")
    parser.add_argument('--lr', type=float, default=0.0002, help="Initial learning rate.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to train.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--pad_len', type=int, default=1200, help="Padd length.")
    parser.add_argument('--results_dir', type=str, default='./results/', help="Path to directory with results and models.")
    parser.add_argument('--ont', type=str, default='molecular_function', help="Ontology.")
    parser.add_argument('--cmap_type', type=str, default='A_ca', help="Contact maps type.")
    parser.add_argument('--model_name', type=str, default='GCN_MF', help="Name of the GCN model.")
    parser.add_argument('--lm_model_name', type=str, help="Name of the LM model.")
    parser.add_argument('--split_fn', type=str,
                        default='/mnt/home/vgligorijevic/Projects/NewMethods/Contact_maps/go_annot/pdb_GO_train_test_split_bc_30.pckl',
                        help="Train-Test split for model's trainig.")
    parser.add_argument('--catalogue', type=str, default="/mnt/ceph/users/vgligorijevic/ContactMaps/data/nr_pdb_chains/catalogue.csv",
                        help="Catalogue with chain to file (in *.npz format) path mapping.")
    parser.add_argument('--train_tfrecord_fn', type=str,
                        default="/mnt/ceph/users/vgligorijevic/ContactMaps/data/Swiss-Model/merged_annot/tfrecords/*_chains_molecular_function_seqid_30_train_EXP-IEA",
                        help="Train tfrecords.")
    parser.add_argument('--valid_tfrecord_fn', type=str,
                        default="/mnt/ceph/users/vgligorijevic/ContactMaps/data/Swiss-Model/merged_annot/tfrecords/*_chains_molecular_function_seqid_30_valid_EXP-IEA",
                        help="Valid tfrecords.")

    args = parser.parse_args()
    print (args)

    train_tfrecord_fn_list = glob.glob(args.train_tfrecord_fn + '*')
    valid_tfrecord_fn_list = glob.glob(args.valid_tfrecord_fn + '*')

    # load annotations
    annot = pickle.load(open(args.split_fn, 'rb'))
    goterms = annot['goterms']
    gonames = annot['gonames']
    Y_train = annot['Y_train']
    if not isinstance(Y_train, np.ndarray):
        Y_train = Y_train.toarray()

    Y_valid = annot['Y_valid']
    if not isinstance(Y_valid, np.ndarray):
        Y_valid = Y_valid.toarray()
    Y_train = np.concatenate([Y_train, Y_valid], axis=0)

    Y_test = annot['Y_test']
    if not isinstance(Y_test, np.ndarray):
        Y_test = Y_test.toarray()
    test_cmap_types = annot['test_cmap_type']
    test_chains = annot['test_pdb_chains']

    ontology = annot['ontology']
    go_idx = np.where(ontology == args.ont)[0]
    output_dim = len(go_idx)

    Y_train = Y_train[:, go_idx]
    Y_test = Y_test[:, go_idx]
    goterms = goterms[go_idx]
    gonames = gonames[go_idx]
    ontology = ontology[go_idx]

    if args.train_tfrecord_fn.find('IEA') > 0:
        Y_train[Y_train == -1] = 1
    else:
        Y_train[Y_train == -1] = 0

    # computing weights for imbalanced go classes
    class_sizes = np.asfarray(Y_train.sum(axis=0))
    mean_class_size = np.mean(class_sizes)
    pos_weights = mean_class_size / class_sizes
    pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights))
    pos_weights = np.concatenate([pos_weights.reshape((len(pos_weights), 1)), pos_weights.reshape((len(pos_weights), 1))], axis=-1)

    print ("### Training model: ", args.model_name, " on ", output_dim, " GO terms.")
    model = DeepFRI(output_dim=output_dim, results_dir=args.results_dir, n_channels=26, gcn_dims=args.gcn_dims,
                    hidd_dims=args.hidden_dims, lr=args.lr, drop=args.dropout, l2_reg=args.l2_reg, lm_model_name=args.lm_model_name)
    model.train(train_tfrecord_fn_list, valid_tfrecord_fn_list, args.model_name, epochs=args.epochs,
                batch_size=args.batch_size, pad_len=args.pad_len, cmap_type=args.cmap_type, class_weight=pos_weights)

    # save models
    model.plot_losses()
    model.save_model()

    # model = load_model(args.results_dir + args.model_name + '.hdf5', custom_objects={'GraphCNN': GraphCNN})
    # print (model.summary())

    # save metadata
    chain2path = load_catalogue(fn=args.catalogue)
    pickle.dump({'goterms': goterms, 'gonames': gonames}, open(args.results_dir + args.model_name + '_metadata.pckl', 'wb'))

    # compute perf on test chains
    print ("### Computing predictions on test set...")
    Y_hat_test = np.zeros_like(Y_test, dtype=float)
    Y_dummy_test = np.zeros_like(Y_test, dtype=float)
    Y_rnd_test = np.zeros_like(Y_test, dtype=float)
    lengths = []
    for i, chain in enumerate(test_chains):
        cmap = np.load(chain2path[chain])
        if args.cmap_type == 'A_ca':
            A = cmap['A_ca_10A']
        elif args.cmap_type == 'A_all':
            A = cmap['A_all_6A']
        elif args.cmap_type == 'A_nbr':
            if 'A_nbr' in cmap:
                A = cmap['A_nbr']
            else:
                A = norm_adj(cmap['A_nbr_bin'])
        else:
            print ('## Wrong key in cmap!')

        S = seq2onehot(str(cmap['sequence']))
        L = cmap['L'].item()
        p = (np.count_nonzero(A) - L)/float(L*(L-1))

        I = np.eye(L)
        A_rnd = rnd_adj(L, p)
        # ##
        S = S.reshape(1, *S.shape)
        A = A.reshape(1, *A.shape)
        A_rnd = A_rnd.reshape(1, *A_rnd.shape)
        I = I.reshape(1, *I.shape)
        # ##
        Y_hat_test[i] = model.predict([A, S])[:, :, 0]
        Y_dummy_test[i] = model.predict([I, S])[:, :, 0]
        Y_rnd_test[i] = model.predict([A_rnd, S])[:, :, 0]
        lengths.append(L)
    lengths = np.asarray(lengths)
    num_pos = Y_train.sum(axis=0)
    pickle.dump({'Y_test': Y_test, 'Y_hat_test': Y_hat_test, 'num_pos': num_pos, 'L_test': lengths,
                'Y_dummy_test': Y_dummy_test, 'Y_rnd_test': Y_rnd_test, 'goterms': goterms,
                 'gonames': gonames, 'chain_ids': test_chains, 'cmap_types': test_cmap_types},
                open(args.results_dir + args.model_name + '_pred_scores.pckl', 'wb'))
