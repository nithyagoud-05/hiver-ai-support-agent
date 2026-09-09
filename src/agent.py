import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SpotifySupportAgent:
    def __init__(self, corpus_df, corpus_tfidf, tfidf_vec, lr_clf):
        self.corpus_df = corpus_df
        self.corpus_tfidf = corpus_tfidf
        self.tfidf = tfidf_vec
        self.lr_model = lr_clf
        self.classes = lr_clf.classes_
        
        self.escalation_intents = {'SUBSCRIPTION_BILLING', 'ACCOUNT_LOGIN', 'UNCLEAR_FEEDBACK'}
        self.grounded_sop_templates = {
            'PLAYBACK_CRASH': "We recommend trying a quick clean reinstall and clearing your app cache (Settings > Storage > Clear Cache). If the issue persists, let us know your device OS version.",
            'OFFLINE_SYNC': "Please make sure your device is connected to active Wi-Fi and toggle Offline Mode off and on in Settings > Playback. Verify you have sufficient storage space available on your device.",
            'SUBSCRIPTION_BILLING': "For billing and subscription security, please head to your account page at spotify.com/account or contact our direct billing team via private DM with your registered email.",
            'ACCOUNT_LOGIN': "To secure your account or reset credentials, please visit spotify.com/password-reset. If you suspect unauthorized access, our security team will assist you via DM.",
            'CATALOG_CONTENT': "Music availability can vary over time due to licensing agreements with rights holders. If an album is greyed out, it may temporarily be unavailable in your region.",
            'PLAYLIST_LIBRARY': "You can recover deleted playlists within 90 days by logging into your account page at spotify.com/account and selecting 'Recover playlists'.",
            'CONNECT_HARDWARE': "Try restarting both your Spotify device and external speaker/Bluetooth unit. Ensure Spotify Connect permissions are enabled in local network settings.",
            'UNCLEAR_FEEDBACK': "Could you let us know what device model and Spotify version you're currently using so we can investigate further?"
        }

    def classify_intent(self, text):
        vec = self.tfidf.transform([text])
        probs = self.lr_model.predict_proba(vec)[0]
        max_idx = np.argmax(probs)
        return self.classes[max_idx], float(probs[max_idx])

    def retrieve_similar_cases(self, text, target_intent, top_k=2):
        vec = self.tfidf.transform([text])
        sims = cosine_similarity(vec, self.corpus_tfidf)[0]
        intent_mask = (self.corpus_df['intent'] == target_intent).values
        if np.sum(intent_mask) >= top_k:
            filtered_indices = np.where(intent_mask)[0]
            top_sub_indices = filtered_indices[np.argsort(sims[filtered_indices])[::-1][:top_k]]
        else:
            top_sub_indices = np.argsort(sims)[::-1][:top_k]
            
        retrieved = []
        for idx in top_sub_indices:
            retrieved.append({
                "historical_customer": self.corpus_df.iloc[idx]['clean_customer'],
                "historical_agent": self.corpus_df.iloc[idx]['clean_agent'],
                "similarity_score": round(float(sims[idx]), 4)
            })
        return retrieved

    def decide_escalation(self, intent, confidence, customer_text):
        if intent in self.escalation_intents:
            return "ESCALATE", f"Intent '{intent}' involves private authentication, payments, or requires user clarification."
        if confidence < 0.40:
            return "ESCALATE", f"Low model confidence ({confidence:.2f}) on intent classification."
        urgent_keywords = ['sue', 'lawyer', 'fraud', 'stolen', 'bank', 'police', 'cancel subscription immediately']
        if any(w in customer_text.lower() for w in urgent_keywords):
            return "ESCALATE", "High customer distress or risk trigger detected in message."
        return "AUTO_HANDLE", f"Routine client-side resolution available with {confidence*100:.1f}% confidence."

    def generate_reply(self, intent, escalation_decision):
        if escalation_decision == "ESCALATE":
            if intent == 'SUBSCRIPTION_BILLING':
                return "We'd like to look into this billing query directly. Please reach out to our team via private DM with your account email or visit spotify.com/account for direct support."
            elif intent == 'ACCOUNT_LOGIN':
                return "To protect your account details, please check spotify.com/password-reset or reach out to us via direct message so we can verify your account securely."
            else:
                return "Could you provide a few more details about your device model, OS, and Spotify app version so our support team can take a closer look?"
        return self.grounded_sop_templates.get(intent, "Please try restarting your app and check for updates.")

    def process(self, customer_message):
        intent, conf = self.classify_intent(customer_message)
        retrieved = self.retrieve_similar_cases(customer_message, intent, top_k=2)
        decision, reason = self.decide_escalation(intent, conf, customer_message)
        reply = self.generate_reply(intent, decision)
        return {
            "intent": intent,
            "confidence": round(conf, 4),
            "escalation": decision,
            "escalation_reason": reason,
            "retrieved_evidence": retrieved,
            "draft_reply": reply
        }
