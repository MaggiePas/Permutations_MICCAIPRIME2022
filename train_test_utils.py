import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, mean_squared_error
import itertools
import numpy as np


def binary_acc(y_pred, y_test, seq2seq):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    if seq2seq:
        correct_results_sum = (y_pred_tag == y_test._base).sum().float()
        acc = correct_results_sum / len(y_test)
    else:
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.unsqueeze(axis=0).shape[0]
    acc = torch.round(acc * 100)

    return acc


# https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model
def l1_regularizer(model, lambda_l1=0.01, weight_or_bias='weight'):
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
            if weight_or_bias in model_param_name:
                lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1


def train_gru(model=None, criterion=None, optimizer=None, max_epochs=30, train_loader=None, val_loader=None, device=None, seq2seq=True, params=None):
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        avg_loss = 0.
        avg_acc = 0.
        counter = 0

        for local_batch, local_labels in train_loader:
            counter += 1
            h = model.init_hidden(batch_size=params['batch_size'])
            h = h.to(device)
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            if seq2seq:
                local_batch = local_batch.squeeze(axis=0)
                local_labels = local_labels.squeeze(axis=0)
                local_batch = local_batch.unsqueeze(axis=1)
                local_labels = local_labels.unsqueeze(axis=1)
            else:
                local_labels = local_labels[:, -1].to(device)

            optimizer.zero_grad()

            h = h.data
            out, h = model(local_batch.float(), h.float())

            loss = criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device))

            acc = binary_acc(out.squeeze(), local_labels, seq2seq)

            total_loss = loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')

            total_loss.backward()
            optimizer.step()
            avg_loss += total_loss.item()
            avg_acc += acc.item()

            if counter % 1000 == 0:
                print(
                    "Epoch {}... Step: {}/{}... Average Loss for Epoch: {}... Accuracy: {}".format(epoch, counter, len(train_loader),
                                                                                         avg_loss / counter, avg_acc / counter))
                if seq2seq:
                    evaluate_all_timesteps_per_subject(model=model, val_loader=val_loader, hidden=h, device=device)
                else:
                    evaluate_last_timestep(model=model, val_loader=val_loader, device=device)

                model.train()
    return model, h


def train_gru_age(model=None, criterion=None, optimizer=None, max_epochs=30, train_loader=None, val_loader=None, device=None,
                  params=None):

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        avg_loss = 0.
        avg_acc = 0.
        counter = 0

        for local_batch, local_labels, local_ages in train_loader:
            counter += 1
            h = model.init_hidden(batch_size=local_batch.shape[0])
            h = h.to(device)
            local_batch, local_labels, local_ages = local_batch.to(device), local_labels.to(device), local_ages.to(device)
            local_batch = local_batch.squeeze(axis=0)
            local_labels = local_labels.squeeze(axis=0)
            local_batch = local_batch.unsqueeze(axis=1)
            local_labels = local_labels.unsqueeze(axis=1)
            local_ages = local_ages.squeeze(axis=0)

            optimizer.zero_grad()

            h = h.data
            out, h, out_age = model(local_batch.float(), h.float())

            loss_score = criterion['score'](out.squeeze().to(device), local_labels.squeeze().float().to(device))
            loss_age = criterion['age'](out_age.squeeze(), local_ages.squeeze().float())

            # The loss is a combination of the BCE and the MSE for the age.
            # I have weighted higher the BCE loss in this case after hyperparameter tuning
            loss = loss_score + 0.2*loss_age

            acc = binary_acc(out.squeeze(), local_labels)

            # I am adding L1 regularization for the weights to minimize overfitting since out dataset is so small
            total_loss = loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')

            total_loss.backward()
            optimizer.step()
            avg_loss += total_loss.item()
            avg_acc += acc.item()

            # Select how often you want your results to be printed during training
            if counter % 6000 == 0:
                print(
                    "Epoch {}... Step: {}/{}... Average Loss for Epoch: {}... Accuracy: {}".format(epoch, counter, len(train_loader),
                                                                                         avg_loss / counter, avg_acc / counter))
                #evaluate_all_timesteps_age(model=model, val_loader=val_loader, hidden=h, device=device)
                # This function returns also the subject-level accuracy and macro accuracy
                evaluate_all_timesteps_age_per_subject(model=model, val_loader=val_loader, hidden=h, device=device)

                model.train()
    return model, h


def evaluate_all_timesteps_age_per_subject(model=None, val_loader=None, hidden=None, device=None):
    y_pred_list = []
    y_pred_ages = []
    y_test = []
    y_test_ages = []
    model.eval()
    subject_acc = []
    subject_bacc_control = []
    subject_bacc_diseased = []
    with torch.no_grad():
        for local_batch, local_labels, local_ages in val_loader:
            h = model.init_hidden(batch_size=local_batch.shape[0])
            h = h.to(device)
            local_batch, local_labels, local_ages = local_batch.to(device), local_labels.to(device), local_ages.to(device)
            local_batch = local_batch.squeeze(axis=0)
            local_labels = local_labels.squeeze(axis=0)
            local_batch = local_batch.unsqueeze(axis=1)
            local_labels = local_labels.unsqueeze(axis=1)
            local_ages = local_ages.squeeze(axis=0)

            h = h.data
            out, h, out_ages = model(local_batch.float(), h.float())
            y_pred_tag = torch.round(torch.sigmoid(out.squeeze()))
            out_ages = out_ages.squeeze()

            # We need to change the labels from an array of arrays to a normal array otherwise the accuracy is not
            # calculated correctly
            label_list = local_labels._base.cpu().numpy().tolist()
            label_list = list(itertools.chain.from_iterable(label_list))

            # Create a list of all the predictions and calculate the subject-level accuracy over all visits
            if len(y_pred_tag.shape) == 0:
                y_pred_list = y_pred_list + [(y_pred_tag.cpu().numpy().tolist())]
                y_pred_ages = y_pred_ages + [(out_ages.cpu().numpy().tolist())]
                subject_acc.append(accuracy_score(label_list, [y_pred_tag.cpu().numpy()]))
            else:
                y_pred_list = y_pred_list + (y_pred_tag.cpu().numpy().tolist())
                y_pred_ages = y_pred_ages + (out_ages.cpu().numpy().tolist())
                subject_acc.append(accuracy_score(label_list, y_pred_tag.cpu().numpy()))

            # Keep the accuracies of control and diseased subjects separately so we can calculate the
            # overall subject-level macro accuracy
            if torch.sum(local_labels) > 0:
                subject_bacc_diseased.append(subject_acc[-1])
            else:
                subject_bacc_control.append(subject_acc[-1])

            y_test = y_test + (local_labels._base.cpu().numpy().tolist())
            y_test_ages = y_test_ages + (local_ages._base.cpu().numpy().tolist())

    y_test = list(itertools.chain.from_iterable(y_test))
    y_test_ages = list(itertools.chain.from_iterable(y_test_ages))
    subject_macro_accuracy = (sum(subject_bacc_diseased)/len(subject_bacc_diseased)
                              + sum(subject_bacc_control)/len(subject_bacc_control))/2
    results_dict = {'subject_accuracy': sum(subject_acc)/len(subject_acc),
                    'subject_macro_accuracy': subject_macro_accuracy,
                    'accuracy': accuracy_score(y_test, y_pred_list),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                    'f1-score': f1_score(y_test, y_pred_list, average='macro'),
                    'mse-age': mean_squared_error(np.array(y_test_ages), np.array(y_pred_ages))}

    # Subject-level means we calculate one score per subject over all visits.
    # That way, subjects with more visits do not influence the results more than subjects with just one visit
    # Afterwards we also calculate the overall results over all visits, regardless of subject

    return results_dict


def evaluate_all_timesteps_per_subject(model=None, val_loader=None, hidden=None, device=None):
    y_pred_list = []
    y_test = []
    model.eval()
    subject_acc = []
    subject_bacc_control = []
    subject_bacc_diseased = []
    with torch.no_grad():
        for local_batch, local_labels in val_loader:
            h = model.init_hidden(batch_size=local_batch.shape[0])
            h = h.to(device)
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            local_batch = local_batch.squeeze(axis=0)
            local_labels = local_labels.squeeze(axis=0)
            local_batch = local_batch.unsqueeze(axis=1)
            local_labels = local_labels.unsqueeze(axis=1)

            h = h.data
            out, h = model(local_batch.float(), h.float())
            y_pred_tag = torch.round(torch.sigmoid(out.squeeze()))

            # We need to change it for an array of arrays to a normal array otherwise the accuracy is not
            # calculated correctly
            patata = local_labels._base.cpu().numpy().tolist()
            patata = list(itertools.chain.from_iterable(patata))

            # Create a list of all the predictions and calculate the subject-level accuracy
            if len(y_pred_tag.shape) == 0:
                y_pred_list = y_pred_list + [(y_pred_tag.cpu().numpy().tolist())]
                subject_acc.append(accuracy_score(patata, [y_pred_tag.cpu().numpy()]))
            else:
                y_pred_list = y_pred_list + (y_pred_tag.cpu().numpy().tolist())
                subject_acc.append(accuracy_score(patata, y_pred_tag.cpu().numpy()))

            # Keep the accuracies of control and diseased subjects separately so we can calculate the
            # overall subject-level macro accuracy
            if torch.sum(local_labels) > 0:
                subject_bacc_diseased.append(subject_acc[-1])
            else:
                subject_bacc_control.append(subject_acc[-1])

            y_test = y_test + (local_labels._base.cpu().numpy().tolist())

    y_test = list(itertools.chain.from_iterable(y_test))
    subject_macro_accuracy = (sum(subject_bacc_diseased)/len(subject_bacc_diseased)
                              + sum(subject_bacc_control)/len(subject_bacc_control))/2
    results_dict = {'subject_accuracy': sum(subject_acc)/len(subject_acc),
                    'subject_macro_accuracy': subject_macro_accuracy,
                    'accuracy': accuracy_score(y_test, y_pred_list),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                    'f1-score': f1_score(y_test, y_pred_list, average='macro')}
    return results_dict


def acc_per_run_dataset_cross_sectional():

    baseline_acc = 0.761871
    baseline_bacc = 0.722522
    baseline_f1 = 0.64044

    return baseline_acc, baseline_bacc, baseline_f1

def acc_per_run(age=False):

    # Accuracies for seq2one model trained on the Google CV
    if age:
        # In this iteration of the study we don't predict age
        # No aces, negative valence, only tabular, multi task, seed 1964, seq2seq
        baseline_subj_acc = 0.0
        baseline_subj_balanced_acc = 0.0
        baseline_acc = 0.0
        baseline_balanced_acc = 0.0
        baseline_f1 = 0.0
    else:
        baseline_subj_acc = 0.874323
        baseline_subj_balanced_acc = 0.745592
        baseline_acc = 0.874323
        baseline_balanced_acc = 0.745592
        baseline_f1 = 0.648194

    return baseline_subj_acc, baseline_subj_balanced_acc, baseline_acc, baseline_balanced_acc, baseline_f1


def evaluate_last_timestep(model=None, val_loader=None, device=None):
    y_pred_list = []
    y_test = []
    model.eval()
    with torch.no_grad():
        # No age
        if len(next(iter(val_loader))) == 2:
            for local_batch, local_labels in val_loader:
                local_batch = local_batch.to(device)
                # local_labels = torch.max(local_labels)
                local_labels = local_labels[:, -1].to(device)

                h = model.init_hidden(batch_size=local_batch.shape[0])
                h = h.to(device)
                out, h = model(local_batch.float(), h.float())
                y_pred_tag = torch.round(torch.sigmoid(out.squeeze()))
                y_pred_list.append(y_pred_tag.cpu().numpy())
                y_test.append(local_labels.cpu().numpy())
        # With age
        elif len(next(iter(val_loader))) == 3:
            for local_batch, local_labels, local_ages in val_loader:
                local_batch = local_batch.to(device)
                # local_labels = torch.max(local_labels)
                local_labels = local_labels[:, -1].to(device)

                h = model.init_hidden(batch_size=local_batch.shape[0])
                h = h.to(device)
                out, h, out_age = model(local_batch.float(), h.float())
                y_pred_tag = torch.round(torch.sigmoid(out.squeeze()))
                y_pred_list.append(y_pred_tag.cpu().numpy())
                y_test.append(local_labels.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_test = [a.squeeze().tolist() for a in y_test]

    # In this case subject accuracy and accuracy are the same since we have only one visit per subject
    results_dict = {'subject_accuracy': accuracy_score(y_test, y_pred_list),
                    'subject_macro_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                    'accuracy': accuracy_score(y_test, y_pred_list),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                    'f1-score': f1_score(y_test, y_pred_list, average='macro')}

    #print(results_dict)
    return results_dict


def train_gru(model=None, criterion=None, optimizer=None, max_epochs=30, train_loader=None, val_loader=None, device=None, seq2seq=True, params=None):
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        avg_loss = 0.
        avg_acc = 0.
        counter = 0

        for local_batch, local_labels in train_loader:
            counter += 1
            h = model.init_hidden(batch_size=params['batch_size'])
            h = h.to(device)
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            if seq2seq:
                local_batch = local_batch.squeeze(axis=0)
                local_labels = local_labels.squeeze(axis=0)
                local_batch = local_batch.unsqueeze(axis=1)
                local_labels = local_labels.unsqueeze(axis=1)
            else:
                local_labels = local_labels[:, -1].to(device)

            optimizer.zero_grad()

            h = h.data
            out, h = model(local_batch.float(), h.float())

            loss = criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device))

            acc = binary_acc(out.squeeze(), local_labels, seq2seq)

            total_loss = loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')

            total_loss.backward()
            optimizer.step()
            avg_loss += total_loss.item()
            avg_acc += acc.item()

            if counter % 1000 == 0:
                print(
                    "Epoch {}... Step: {}/{}... Average Loss for Epoch: {}... Accuracy: {}".format(epoch, counter, len(train_loader),
                                                                                         avg_loss / counter, avg_acc / counter))
                if seq2seq:
                    evaluate_all_timesteps_per_subject(model=model, val_loader=val_loader, hidden=h, device=device)
                else:
                    evaluate_last_timestep(model=model, val_loader=val_loader, device=device)

                model.train()
    return model, h
