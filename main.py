import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from data_loader import SingleSubjectDataset, MultiSubjectDataset
from torchvision import datasets
from torchvision import transforms
from model import DANN
import argparse

def test(model, dataloader, device):
    
    alpha = 0

    model = model.eval()

    model = model.to(device)
    
    with torch.no_grad():
       
        n_total = 0
        n_correct = 0

        for target_signal, target_label, target_domain in dataloader :


            batch_size = len(target_label)

            target_signal = target_signal.to(device, dtype=torch.float)
            target_label = target_label.to(device, dtype=torch.long)


            class_output, _ = model(input_data=target_signal, alpha=alpha)
            pred = class_output.data.max(1, keepdim=True)[1]


            n_correct += pred.eq(target_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size


        accu = n_correct.data.numpy() * 1.0 / n_total

        return accu


if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description='Domain adaptation for EEG motor imagery')

    parser.add_argument('--dataroot', default='dataset/2a/', type=str,
                        help='dataset root')

    parser.add_argument('--batch', default=20, type=int,
                        help='Batch size for training. Evaluation batch size will be twice as big.')

    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Learning rate for Adam optimizer')

    parser.add_argument('--epochs', default=200, type=int,
                        help='Number of epochs to train for')

    parser.add_argument('--annotated', default=0.0, type=float,
                        help='The default ratio of annotated examples from the target domain to learn from.')

    parser.add_argument('--features', default='attention', type=str,
                        help='Pick a feature generator for DANN. "concat" just processes the signal, "attention" is equivalent to PSTSA, double_attention add additional attention layer to PSTSA')

    parser.add_argument('--dropout', default='store_true', 
                        help='Toggle to use dropout')

    parser.add_argument('--target_weight', default=1.0, type=float,
                        help='Increases lr for target domain on classification to make training on it more substantial')


    opt = parser.parse_args()

    lr = opt.lr
    batch_size = opt.batch
    batch_size_eval = batch_size*2
    n_epoch = opt.epochs
    dataroot = opt.dataroot
    annotated_percentage = opt.annotated
    feature_mode = opt.features
    use_dropout = opt.dropout
    target_weight = opt.target_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    file_names=["A01T", "A02T", "A03T", "A04T", "A05T", "A06T", "A07T", "A08T", "A09T"]

    file_names_eval=["A01E", "A02E", "A03E", "A04E", "A05E", "A06E", "A07E", "A08E", "A09E"]

    domain_ids = [0,1,2,3,4,5,6,7,8]


    random_id = np.random.randint(2)

    target_filename = file_names[random_id]
    source_filenames = [f for f in file_names if f != target_filename]

    target_id = domain_ids[random_id]
    source_ids = [i for i in domain_ids if i != target_id]

    target_filename_eval = file_names_eval[random_id]
    source_filenames_eval = [f for f in file_names_eval if f != target_filename_eval]

    dataset_target = SingleSubjectDataset(file_name=target_filename, root_dir=dataroot, domain_id=target_id, annotated_percentage=annotated_percentage)

    dataset_source = MultiSubjectDataset(file_names=source_filenames, root_dir=dataroot, domain_ids=source_ids)

    dataset_target_eval = SingleSubjectDataset(file_name=target_filename_eval, root_dir=dataroot, domain_id=target_id)

    dataset_source_eval = MultiSubjectDataset(file_names=source_filenames_eval, root_dir=dataroot, domain_ids=source_ids)


    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)


    dataloader_source_eval = torch.utils.data.DataLoader(
        dataset=dataset_source_eval,
        batch_size=batch_size_eval,
        shuffle=True,
        num_workers=8)

    dataloader_target_eval = torch.utils.data.DataLoader(
        dataset=dataset_target_eval,
        batch_size=batch_size_eval,
        shuffle=True,
        num_workers=8)


    model = DANN(feature_mode = feature_mode, use_dropout = use_dropout)


    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    model.to(device)
    loss_class.to(device)
    loss_domain.to(device)

    for p in model.parameters():
        p.requires_grad = True

    # training
    best_accu_t = 0.0
    data_target_iter = iter(dataloader_target)

    for epoch in range(n_epoch):

        len_dataloader = len(dataloader_source)
        data_source_iter = iter(dataloader_source)
        
        model.train()


        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            data_source = data_source_iter.next()
            source_signal, source_label, domain_label = data_source

            model.zero_grad()
            #batch_size = len(s_label)

            #domain_label = torch.zeros(batch_size).long()

            source_signal = source_signal.to(device, dtype=torch.float)
            source_label = source_label.to(device, dtype=torch.long)
            domain_label = domain_label.to(device, dtype=torch.long)

            class_y, domain_y = model(input_data=source_signal, alpha=alpha)


            source_class_loss = loss_class(class_y, source_label)
            source_domain_loss = loss_domain(domain_y, domain_label)

            

            try:
                data_target = data_target_iter.next()
            except StopIteration:
                # restart the iterator if the previous iterator is exhausted.
                data_target_iter = iter(dataloader_target)
                data_target = data_target_iter.next()


            target_signal, target_label, domain_label = data_target


            target_signal = target_signal.to(device=device, dtype=torch.float)
            domain_label = domain_label.to(device=device, dtype=torch.long)

            class_y, domain_y = model(input_data=target_signal, alpha=alpha)

            target_domain_loss = loss_domain(domain_y, domain_label)
            
            loss = target_domain_loss + source_domain_loss + source_class_loss

            target_class_loss = 0

            if torch.any (target_label >= 0):
                target_class_loss = loss_class(class_y[target_label >= 0], target_label[target_label >= 0])
                loss += target_weight * target_class_loss
        
            loss.backward()
            optimizer.step()

        print('\r epoch: %d, \n loss source classification: %f, \n loss source domain : %f, \n loss target classification %f, \n loss target domain: %f' \
            % (epoch, source_class_loss, source_domain_loss, target_class_loss, target_domain_loss)) 

         
        print('\n')
        source_acc = test(model, dataloader_source_eval, device)
        print('Evaluation accuracy on the source domain: %f' %  source_acc)
        target_acc = test(model, dataloader_target_eval, device)
        print('Evaluation accuracy on the target domain: %f' % target_acc)
        if target_acc > best_accu_t:
            best_source_acc = source_acc
            best_target_acc = target_acc
            torch.save(model, 'models/best_model.pth')

    print('============ Summary ============= \n')
    print('Best evaluation accuracy on the source dataset' % best_accu_s)
    print('Accuracy of the %s dataset: %f' %best_accu_t)
    print('Corresponding model was save in ' + model_root + '/mnist_mnistm_model_epoch_best.pth')