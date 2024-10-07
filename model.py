import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

class RelationExtractionDataset(Dataset):
    def __init__(self, ids, texts, entity_pairs, labels, tokenizer, max_len):
        self.ids = ids
        self.texts = texts
        self.entity_pairs = entity_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        entity_pair = self.entity_pairs[item]
        label = self.labels[item]
        id = self.ids[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'entity_pair': torch.tensor(entity_pair, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'id': torch.tensor(id, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size, eval=False):
    ds = RelationExtractionDataset(
        ids=df.id.to_numpy(),
        texts=df.text.to_numpy(),
        entity_pairs=df.entity_pairs.to_numpy(),
        labels=df.labels.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    ) if not eval else RelationExtractionDataset(
        ids=df.id.to_numpy(),
        texts=df.text.to_numpy(),
        entity_pairs=df.entity_pairs.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
    )

def preprocess_data(data, eval=False):
    texts = []
    entity_pairs = []
    entity_pairs_ids = []
    labels = []
    ids = []

    for i, row in data.iterrows():
        ids.append(row['id'])
        text = row['text']
        entities = json.loads(row['entities'])
        relations = json.loads(row['relations'])

        """ for rel in relations:
            entity_1 = entities[rel[0]]
            entity_2 = entities[rel[2]]
            label = rel[1]

            # Assuming entities are already in (start, end) format and entity type is needed
            entity_1_pos = [entity_1['mentions'][0]['start'], entity_1['mentions'][0]['end']]
            entity_2_pos = [entity_2['mentions'][0]['start'], entity_2['mentions'][0]['end']]

            texts.append(text)
            entity_pairs.append([entity_1_pos, entity_2_pos])
            labels.append(label) """
        
        for ent1 in entities:
            for ent2 in entities:
                if ent1 != ent2:
                    print(ent1)
                    entity_1 = entities[ent1]
                    entity_2 = entities[ent2]
                    # another idea to test: add all relations of the mentions (not only the first one) then group by entity id after relations extraction
                    entity_1_pos = [entity_1['mentions'][0]['start'], entity_1['mentions'][0]['end']]
                    entity_2_pos = [entity_2['mentions'][0]['start'], entity_2['mentions'][0]['end']]

                    texts.append(text)
                    entity_pairs.append([entity_1_pos, entity_2_pos])
                    entity_pairs_ids.append([ent1['id'], ent2['id']])
                    # check if there is a relation between the two entities, relations are like: [1, ""INITIATED"", 9]
                    if eval:
                        found_relation = False
                        for rel in relations:
                            if rel[0] == ent1['id'] and rel[2] == ent2['id']:
                                labels.append(rel[1])
                                found_relation = True
                                break
                        if not found_relation:
                            labels.append('NO_RELATION')

    return ids, texts, entity_pairs, entity_pairs_ids, labels

def train_model(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        # cal_loss = loss_fn(preds, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))
    print(f'F1 score: {f1_score(all_labels, all_preds, average="weighted")}')
    print(f'Precision score: {precision_score(all_labels, all_preds, average="weighted")}')
    print(f'Recall score: {recall_score(all_labels, all_preds, average="weighted")}')
    print(f'Accuracy score: {accuracy_score(all_labels, all_preds)}')

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0
    all_preds = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            # labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            # correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())


    return correct_predictions.double() / n_examples, np.mean(losses)

def main():
    # Load data from a CSV or any other format into a pandas DataFrame
    data = pd.read_csv('train/train.csv')
    ids, texts, entity_pairs, entity_pairs_ids, labels = preprocess_data(data)

    df=pd.DataFrame({'id': ids,'text':texts, 'entity_pairs':entity_pairs, 'labels':labels})
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # set torch seed
    torch.manual_seed(313)
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=len(set(labels)))

    MAX_LEN = 512
    BATCH_SIZE = 8
    EPOCHS = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    # total_steps = len(data_loader) * EPOCHS

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(EPOCHS):
        train_acc, train_loss = train_model(
            model,
            data_loader,
            loss_fn,
            optimizer,
            device,
            len(data)
        )

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f'Train loss {train_loss} accuracy {train_acc}')

    print("Training complete.")

    # Saving model
    model.save_pretrained('./relation_extraction_model')
    tokenizer.save_pretrained('./relation_extraction_tokenizer')
    test_data = pd.read_csv('test/test_01-07-2024.csv')
    ids, test_texts, test_entity_pairs, entity_pairs_ids, _ = preprocess_data(test_data, eval=True)
    test_df = pd.DataFrame({'id': ids,'text':test_texts, 'entity_pairs':test_entity_pairs})
    test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    test_acc, test_loss = eval_model( model, test_data_loader, loss_fn, device, len(test_data))
    print(f'Test accuracy {test_acc}')
    print(f'Test loss {test_loss}')

if __name__ == "__main__":
    main()
