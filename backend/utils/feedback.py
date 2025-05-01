import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

class FeedbackManager:
    def __init__(self, feedback_dir: str = "feedback_data"):
        self.feedback_dir = feedback_dir
        os.makedirs(feedback_dir, exist_ok=True)
        self.feedback_file = os.path.join(feedback_dir, 'user_feedback.json')
        self.feedback_data = self._load_feedback()

    def _load_feedback(self) -> List[Dict]:
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_feedback(self):
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)

    def add_feedback(self,
                    question: str,
                    response: str,
                    rating: int,
                    comments: Optional[str] = None,
                    user_id: Optional[str] = None) -> str:
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        feedback_id = str(uuid.uuid4())

        feedback_entry = {
            "id": feedback_id,
            "user_id": user_id or "anonymous",
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": response,
            "rating": rating,
            "comments": comments
        }

        self.feedback_data.append(feedback_entry)
        self._save_feedback()
        return feedback_id

    def get_feedback(self, feedback_id: str) -> Optional[Dict]:
        for entry in self.feedback_data:
            if entry["id"] == feedback_id:
                return entry
        return None

    def get_all_feedback(self) -> List[Dict]:
        return self.feedback_data

    def get_feedback_summary(self) -> Dict[str, Any]:
        if not self.feedback_data:
            return {
                "total_entries": 0,
                "average_rating": 0,
                "rating_distribution": {
                    "1": 0, "2": 0, "3": 0, "4": 0, "5": 0
                }
            }

        total = len(self.feedback_data)
        ratings = [entry["rating"] for entry in self.feedback_data]
        avg_rating = sum(ratings) / total

        rating_counts = {str(i): 0 for i in range(1, 6)}
        for rating in ratings:
            rating_counts[str(rating)] += 1

        return {
            "total_entries": total,
            "average_rating": avg_rating,
            "rating_distribution": rating_counts
        }

    def export_feedback_csv(self, output_path: str) -> str:
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow([
                'ID', 'User ID', 'Timestamp', 'Question',
                'Response', 'Rating', 'Comments'
            ])

            for entry in self.feedback_data:
                writer.writerow([
                    entry['id'],
                    entry['user_id'],
                    entry['timestamp'],
                    entry['question'],
                    entry['response'],
                    entry['rating'],
                    entry['comments'] or ''
                ])

        return output_path