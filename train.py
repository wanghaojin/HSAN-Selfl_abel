import numpy
import torch

from utils import *
from tqdm import tqdm
from torch import optim
from setup import setup_args
from model import hard_sample_aware_network
from sinkhorn_knopp import SinkhornKnopp
from center import get_center
from model import Prototypes
def cross_entropy_loss(preds, targets, temperature=0.1):
    preds = F.log_softmax(preds / temperature, dim=-1)
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)


if __name__ == '__main__':

    # for dataset_name in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
    for dataset_name in ["cora"]:

        # setup hyper-parameter
        args = setup_args(dataset_name)

        # record results
        file = open("result.csv", "a+")
        print(args.dataset, file=file)
        print("ACC,   NMI,   ARI,   F1", file=file)
        file.close()
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        # ten runs with different random seeds
        for args.seed in range(args.runs // 5):
            # record results

            # fix the random seed
            setup_seed(args.seed)

            # load graph data
            X, y, A, node_num, cluster_num = load_graph_data(dataset_name, show_details=False)

            # apply the laplacian filtering
            X_filtered = laplacian_filtering(A, X, args.t)

            # test
            # print('*****************')
            # print(X_filtered.shape)
            # print(X_filtered)
            # print(y.shape)
            # print(cluster_num)
            # print('*****************')
            # assert 0
            #args.acc, args.nmi, args.ari, args.f1, y_hat, center = phi(X_filtered, y, cluster_num)
            # sinkhorn_knopp = SinkhornKnopp()
            # model_pro= Prototypes(X_filtered.shape[1],7)
            # X_label = model_pro(X_filtered)
            # predict_label = torch.argmax(sinkhorn_knopp(X_label),dim=1).numpy()
            # print('predict_label',predict_label)
            # centers = get_center(X_filtered,predict_label,7)
            # print('y',y.shape)
            # print('predict_label',predict_label.shape)
            # args.acc, args.nmi, args.ari, args.f1 = eva(y, predict_label, show_details=False)

            # print('X',X.shape)
            # print('A',A.size())
            # print('X_filtered',X_filtered.size())
            # build our hard sample aware network
            HSAN = hard_sample_aware_network(
                input_dim=X.shape[1], hidden_dim=args.dims, act=args.activate, n_num=node_num, cluster_num=7)

            # adam optimizer
            optimizer = optim.Adam(HSAN.parameters(), lr=args.lr)

            # positive and negative sample pair index matrix
            mask = torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)

            # load data to device
            A, HSAN, X_filtered, mask = map(lambda x: x.to(args.device), (A, HSAN, X_filtered, mask))

            sinkhorn_knopp = SinkhornKnopp()
            # training
            for epoch in tqdm(range(600), desc="training..."):
                # train mode
                HSAN.train()

                # encoding with Eq. (3)-(5)
                Z1, Z2, E1, E2, p1, p2 = HSAN(X_filtered, A)
                Q1 = sinkhorn_knopp(p1)
                Q2 = sinkhorn_knopp(p2)
                # print('Q1',Q1.size())
                loss = cross_entropy_loss(p1, Q2) + cross_entropy_loss(p2, Q1)

                                # calculate comprehensive similarity by Eq. (6)
                # S = comprehensive_similarity(Z1, Z2, E1, E2, HSAN.alpha)
                #
                #                  # calculate hard sample aware contrastive loss by Eq. (10)-(11)
                # total_loss = hard_sample_aware_infoNCE(S, mask, HSAN.pos_neg_weight, HSAN.pos_weight, node_num) + loss

                # optimization
                loss.backward()
                optimizer.step()

                # testing and update weights of sample pairs
                if epoch % 10 == 0:
                    # evaluation mode
                    HSAN.eval()

                    # encoding
                    Z1, Z2, E1, E2, p1, p2 = HSAN(X_filtered, A)
                    # linear
                    Q1 = sinkhorn_knopp(p1)
                    Q2 = sinkhorn_knopp(p2)
                    Q = torch.argmax((Q1+Q2)/2,dim=1)
                    # calculate comprehensive similarity by Eq. (6)
                    # S = comprehensive_similarity(Z1, Z2, E1, E2, HSAN.alpha)
                    centers = get_center(X_filtered.detach().cpu().numpy(),Q.detach().cpu().numpy(),7)
                    # fusion and testing
                    # Z = (Z1 + Z2) / 2

                    # print('*****************')
                    # print(Z.shape)
                    # print(Z)
                    # print(y.shape)
                    # print(cluster_num)
                    # print('*****************')

                    # acc, nmi, ari, f1, P, center = phi(Z.detach(), y, cluster_num)
                    # print('Z',Z.shape[1])


                    # model = Prototypes(Z.shape[1],7)
                    # X_label = model(Z.cpu())
                    # print('X_label',X_label.shape)
                    # predict_label = torch.argmax(sinkhorn_knopp(X_label), dim=1).numpy()
                    # print('predict_label',predict_label.shape)
                    # print('X_filtered',X_filtered.shape)
                    # print('centers',centers.shape)
                    print('y', y.shape)
                    print('Q', Q.shape)
                    acc, nmi, ari, f1 = eva(y,Q.detach().cpu().numpy(), show_details=False)
                    # print('Z',Z.shape)
                    # select high confidence samples
                    # print(Z.device)
                    # print(center.device)
                    # H, H_mat = high_confidence(Z.cpu(), centers)
                    #
                    # # calculate new weight of sample pair by Eq. (9)
                    # M, M_mat = pseudo_matrix(predict_label, S, node_num)

                    # update weight
                    # HSAN.pos_weight[H] = M[H].data
                    # HSAN.pos_neg_weight[H_mat] = M_mat[H_mat].data

                    # recording
                    if acc >= args.acc:
                        args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1

            print("Training complete")

            # record results
            file = open("result.csv", "a+")
            print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(args.acc, args.nmi, args.ari, args.f1), file=file)
            file.close()
            acc_list.append(args.acc)
            nmi_list.append(args.nmi)
            ari_list.append(args.ari)
            f1_list.append(args.f1)

        # record results
        acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
        file = open("result.csv", "a+")
        print("{:.2f}, {:.2f}".format(acc_list.mean(), acc_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(nmi_list.mean(), nmi_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(ari_list.mean(), ari_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(f1_list.mean(), f1_list.std()), file=file)
        file.close()
