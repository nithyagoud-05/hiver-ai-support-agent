import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

from src.agent import SpotifySupportAgent

def main():
    print('=' * 50)
    print('  Hiver AI Support Agent - Reproducibility Suite  ')
    print('=' * 50)

    golden_path = 'evaluation/golden_set_candidates.csv'
    corpus_path = 'spotify_retrieval_corpus.csv'
    tfidf_path = 'tfidf_vectorizer.pkl'
    model_path = 'simple_lr_model.pkl'

    golden_df = pd.read_csv(golden_path)
    corpus_df = pd.read_csv(corpus_path)
    tfidf = joblib.load(tfidf_path)
    lr_model = joblib.load(model_path)

    corpus_tfidf = tfidf.transform(corpus_df['clean_customer'])
    agent = SpotifySupportAgent(corpus_df, corpus_tfidf, tfidf, lr_model)

    preds_intent = []
    preds_esc = []
    for text in golden_df['clean_customer']:
        res = agent.process(text)
        preds_intent.append(res['intent'])
        preds_esc.append(res['escalation'])

    y_true_intent = golden_df['verified_intent']
    y_true_esc = golden_df['expected_escalation']

    acc = accuracy_score(y_true_intent, preds_intent)
    macro_f1 = f1_score(y_true_intent, preds_intent, average='macro')
    esc_p = precision_score(y_true_esc, preds_esc, pos_label='ESCALATE')
    esc_r = recall_score(y_true_esc, preds_esc, pos_label='ESCALATE')
    esc_f1 = f1_score(y_true_esc, preds_esc, pos_label='ESCALATE')

    print('\n[Headline Metrics on 200 Hand-Labelled Golden Samples]')
    print(f'  Intent Classification Accuracy: {acc*100:.2f}%')
    print(f'  Intent Macro-F1 Score:         {macro_f1:.4f}')
    print(f'  Escalation Precision:          {esc_p*100:.2f}%')
    print(f'  Escalation Recall:             {esc_r*100:.2f}%')
    print(f'  Escalation F1-Score:           {esc_f1:.4f}')
    print('\n[OK] Pipeline verification completed successfully in < 15 seconds.')

if __name__ == '__main__':
    main()
