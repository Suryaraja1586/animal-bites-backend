import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os, json
import dotenv
from typing import Optional, List, Dict, Any
from typing import List, Dict, Any
import logging
from pymongo import MongoClient
from typing import List, Dict, Any
import os
from datetime import datetime
from flask import session
from bson import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_firebase_config():
    """Fetch and validate Firebase service account credentials."""
    # Ensure .env is loaded
    dotenv.load_dotenv()
    firebase_key = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    
    if not firebase_key:
        logger.error("FIREBASE_SERVICE_ACCOUNT_KEY not found in environment variables.")
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY not found in environment variables.")
    
    logger.info("Retrieved FIREBASE_SERVICE_ACCOUNT_KEY from environment.")
    
    # Handle JSON string
    try:
        # Strip whitespace and normalize newlines for Windows
        firebase_key = firebase_key.strip().replace('\r\n', '\n')
        config = json.loads(firebase_key)
        logger.info("Successfully parsed FIREBASE_SERVICE_ACCOUNT_KEY as JSON string.")
        
        # Validate required fields
        required_fields = ["type", "project_id", "private_key", "client_email", "client_id"]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            logger.error(f"Invalid Firebase config: Missing fields {missing_fields}")
            raise ValueError(f"Invalid Firebase config: Missing fields {missing_fields}")
        
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse FIREBASE_SERVICE_ACCOUNT_KEY as JSON: {e}")
        logger.debug(f"Problematic JSON string (first 100 chars): {firebase_key[:100]}...")
        raise ValueError(f"Failed to parse FIREBASE_SERVICE_ACCOUNT_KEY as JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in get_firebase_config: {e}")
        raise

def initialize_firebase():
    """Initialize Firebase Firestore client."""
    if firebase_admin._apps:
        logger.info("Firebase app already initialized.")
        return firestore.client()
    try:
        firebase_config = get_firebase_config()
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase initialized successfully.")
        # Test connection
        test_doc = db.collection("DOCTOR").document("1").get()
        logger.info(f"Firebase connection test: Document exists: {test_doc.exists}")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        raise

try:
    db = initialize_firebase()
except Exception as e:
    logger.error(f"Initial Firebase setup failed: {e}")
    raise

# Helper function to check if a conversation is casual
def is_casual_conversation(question, answer):
    """Check if a conversation is casual/unnecessary"""
    casual_patterns = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "thank you", "thanks", "bye", "goodbye", "ok", "okay",
        "yes", "no", "sure", "great", "awesome", "cool", "nice", "fine",
        "what's up", "how's it going", "see you", "take care", "good night",
        "good day", "how do you do", "pleased to meet you", "nice to meet you"
    ]
    
    question_lower = question.lower().strip()
    if any(pattern in question_lower for pattern in casual_patterns):
        return True
    
    if len(question.strip()) < 10:
        return True
    
    answer_lower = answer.lower().strip()
    greeting_responses = [
        "hello", "hi there", "good morning", "good afternoon", "good evening",
        "how can i help", "nice to meet you", "pleased to meet you",
        "thank you for", "you're welcome", "no problem"
    ]
    
    if any(response in answer_lower for response in greeting_responses):
        return True
    
    if len(answer.strip()) < 20:
        return True
    
    return False

# Helper function to check for duplicate questions
def is_duplicate_question(question, existing_questions):
    """Check if a question is a duplicate or very similar to existing ones"""
    question_lower = question.lower().strip()
    
    for existing in existing_questions:
        existing_lower = existing.lower().strip()
        # Simple similarity check - can be enhanced with more sophisticated methods
        if question_lower == existing_lower:
            return True
        # Check if one question contains most words of another (simple similarity)
        q_words = set(question_lower.split())
        e_words = set(existing_lower.split())
        if len(q_words) > 3 and len(e_words) > 3:
            intersection = q_words.intersection(e_words)
            if len(intersection) >= min(len(q_words), len(e_words)) * 0.9:
                return True
    return False

def save_unanswered_question(question_english: str):
    """Save unanswered question with timestamp."""
    logger.info(f"Attempting to save unanswered question: {question_english}")
    
    # Temporarily disable filters for testing
    # if is_casual_conversation(question_english, ""):
    #     logger.info(f"Skipping casual question: {question_english}")
    #     return
    
    doctor_doc_ref = db.collection("DOCTOR").document("1")
    doc = doctor_doc_ref.get()
    data = doc.to_dict() if doc.exists else {}
    
    # Ensure qn array exists
    if not data:
        logger.info("DOCTOR/1 document missing, creating new one.")
        data = {"qn": [], "ans": {}}
        doctor_doc_ref.set(data)
    
    # Get existing questions for duplicate check
    existing_questions = [q.get("question", "") for q in data.get("qn", [])]
    
    # Temporarily disable duplicate check for testing
    # if is_duplicate_question(question_english, existing_questions):
    #     logger.info(f"Skipping duplicate question: {question_english}")
    #     return
    
    question_obj = {
        "question": question_english,
        "timestamp": datetime.now(),
        "status": "pending"
    }
    try:
        doctor_doc_ref.update({"qn": firestore.ArrayUnion([question_obj])})
        logger.info(f"Successfully saved unanswered question: {question_english}")
    except Exception as e:
        logger.error(f"Failed to save unanswered question: {e}")
        raise

def save_user_interaction(question_english: str, answer_english: str, user_session_id: Optional[str] = None):
    """Save user interaction and filter out casual conversations"""
    
    # Don't save casual conversations
    if is_casual_conversation(question_english, answer_english):
        print(f"DEBUG: Skipping casual conversation: {question_english}")
        return
    
    fallback_responses = [
        "Sorry, but I specialize in answering questions related to animal bites",
        "An internal error occurred"
    ]
    
    if any(fallback in answer_english for fallback in fallback_responses):
        print(f"DEBUG: Skipping fallback response: {question_english}")
        return
    
    interaction_data = {
        "question": question_english,
        "answer": answer_english,
        "timestamp": datetime.now(),
        "session_id": user_session_id or "anonymous",
        "status": "answered"
    }
    
    unanswered_indicators = [
        "doctor has been notified",
        "doctor will be notified", 
        "check back in a few days",
        "unable to answer your question"
    ]
    
    # Check if this interaction was forwarded to doctor
    if any(indicator in answer_english.lower() for indicator in unanswered_indicators):
        interaction_data["status"] = "forwarded_to_doctor"
        
        # REMOVED: The duplicate save_unanswered_question call
        # The question is already saved in the main processing logic in app.py
        print(f"DEBUG: Question marked as forwarded to doctor: {question_english}")
    
    db.collection("user").add(interaction_data)
    print(f"DEBUG: Saved user interaction: {question_english}")

# -------- New helpers for dashboard --------


def get_unanswered_questions(limit: int = 50) -> List[Dict[str, Any]]:
    """Get unanswered questions from DOCTOR document."""
    logger.info("Fetching unanswered questions...")
    try:
        doctor_doc_ref = db.collection("DOCTOR").document("1")
        doc = doctor_doc_ref.get()
        if not doc.exists:
            logger.warning("DOCTOR/1 document does not exist.")
            return []
        
        data = doc.to_dict()
        questions = data.get("qn", [])
        pending = [
            {
                "id": f"qn_{i}",
                "question": q["question"],
                "timestamp": q["timestamp"].isoformat() if hasattr(q["timestamp"], "isoformat") else str(q["timestamp"])
            }
            for i, q in enumerate(questions)
            if q.get("status") == "pending"
        ]
        logger.info(f"Found {len(pending)} unanswered questions.")
        return pending[:limit]
    except Exception as e:
        logger.error(f"Error fetching unanswered questions: {e}")
        return []
    
def submit_answer(question: str, answer: str) -> bool:  # Changed return type to bool
    """Submit answer and update question status"""
    try:
        logger.info(f"Starting answer submission for question: {question}")
        
        # === STORAGE LOCATION 1: Update DOCTOR Document in Firestore ===
        doctor_ref = db.collection("DOCTOR").document("1")
        doc = doctor_ref.get()
        data = doc.to_dict() if doc.exists else {}
        
        # Add to answers dictionary
        ans_dict = data.get("ans", {}) or {}
        ans_dict[question] = answer
        logger.info(f"Updated answers dictionary with {len(ans_dict)} total answers")
        
        # Update question status from "pending" to "answered"
        qn_list = data.get("qn", []) or []
        updated_qn_list = []
        question_found = False
        
        for q in qn_list:
            if isinstance(q, dict):
                if q.get("question") == question:
                    q["status"] = "answered"
                    q["answered_at"] = datetime.now()
                    question_found = True
                    logger.info(f"Updated question status to 'answered'")
                updated_qn_list.append(q)
            else:
                # Handle old format
                if q == question:
                    updated_qn_list.append({
                        "question": q,
                        "timestamp": datetime.now(),
                        "status": "answered",
                        "answered_at": datetime.now()
                    })
                    question_found = True
                    logger.info(f"Converted old format question to new format")
                else:
                    updated_qn_list.append({
                        "question": q,
                        "timestamp": datetime.now(),
                        "status": "pending"
                    })
        
        if not question_found:
            logger.warning(f"Question not found in pending list: {question}")
            # Still continue with the process
        
        # Update DOCTOR document
        doctor_ref.set({
            "ans": ans_dict,
            "qn": updated_qn_list
        }, merge=True)
        logger.info("Successfully updated DOCTOR document")

        # === STORAGE LOCATION 2: Solved Questions Collection ===
        solved_qa_data = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(),
            "source": "dashboard_submit",
            "status": "active"
        }
        solved_doc_ref = db.collection("solved_questions").add(solved_qa_data)
        logger.info(f"Added to solved_questions collection with ID: {solved_doc_ref[1].id}")

        # === STORAGE LOCATION 3: MongoDB Vector Database ===
        mongodb_success = False
        try:
            # Check if embedding module exists and has the required function
            try:
                import embedding
                if hasattr(embedding, 'store_question_answer'):
                    embedding.store_question_answer(question, answer)
                    mongodb_success = True
                    logger.info("Successfully stored Q&A in MongoDB")
                else:
                    logger.warning("store_question_answer function not found in embedding module")
            except ImportError:
                logger.warning("embedding module not found - MongoDB storage skipped")
            except Exception as e:
                logger.error(f"Error storing in MongoDB: {e}")
        except Exception as e:
            logger.error(f"Unexpected error with MongoDB storage: {e}")
            # Don't let MongoDB errors break the whole process

        # === STORAGE LOCATION 4: User Interaction Log ===
        interaction_data = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(),
            "session_id": "dashboard",
            "status": "answered_by_doctor"
        }
        user_doc_ref = db.collection("user").add(interaction_data)
        logger.info(f"Added to user collection with ID: {user_doc_ref[1].id}")
        
        logger.info(f"Successfully completed answer submission for: {question}")
        logger.info(f"MongoDB storage: {'Success' if mongodb_success else 'Skipped/Failed'}")
        
        return True  # Return True on successful completion
        
    except Exception as e:
        logger.error(f"Failed to submit answer for '{question}': {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise  # Re-raise the exception to be caught by the API endpoint

def get_user_queries(limit: int = 100) -> List[Dict[str, Any]]:
    """Get user queries excluding casual conversations"""
    try:
        q = db.collection("user").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
        rows = []
        for doc in q.stream():
            d = doc.to_dict()
            question = d.get("question", "")
            answer = d.get("answer", "")
            
            # Filter out casual conversations
            if not is_casual_conversation(question, answer):
                ts = d.get("timestamp")
                d["id"] = doc.id
                d["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                rows.append(d)
        
        print(f"DEBUG: Found {len(rows)} user queries")
        return rows
        
    except Exception as e:
        print(f"ERROR: Failed to get user queries: {e}")
        return []

def get_daily_stats() -> Dict[str, int]:
    """Get statistics for today"""
    try:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Count resolved questions today
        resolved_today = 0
        try:
            query = db.collection("solved_questions").where("timestamp", ">=", today).where("status", "==", "active")
            resolved_today = len(list(query.stream()))
        except Exception as e:
            print(f"DEBUG: Error getting daily stats: {e}")
        
        # Get total queries and pending questions
        total_queries = len(get_user_queries(500))
        pending_questions = len(get_unanswered_questions())
        
        stats = {
            "resolved_today": resolved_today,
            "total_queries": total_queries,
            "pending_questions": pending_questions
        }
        
        print(f"DEBUG: Stats - {stats}")
        return stats
        
    except Exception as e:
        print(f"ERROR: Failed to get daily stats: {e}")
        return {"resolved_today": 0, "total_queries": 0, "pending_questions": 0}

def add_question_answer(question: str, answer: str) -> None:
    """Add new Q&A pair"""
    try:
        # Store in MongoDB
        try:
            from embedding import store_question_answer
            store_question_answer(question, answer)
            print(f"DEBUG: Successfully stored new Q&A in MongoDB: {question}")
        except Exception as e:
            print(f"ERROR: Failed to store in MongoDB: {e}")
        
        # Update DOCTOR document
        doctor_ref = db.collection("DOCTOR").document("1")
        doc = doctor_ref.get()
        data = doc.to_dict() if doc.exists else {}
        ans_dict = data.get("ans", {}) or {}
        ans_dict[question] = answer
        doctor_ref.set({"ans": ans_dict}, merge=True)
        
        # Store in solved questions collection
        solved_qa_data = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(),
            "source": "dashboard_manual",
            "status": "active"
        }
        db.collection("solved_questions").add(solved_qa_data)
        
        # Save interaction record
        db.collection("user").add({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(),
            "session_id": "dashboard_manual",
            "status": "answered_by_doctor"
        })
        
        print(f"DEBUG: Successfully added new Q&A: {question}")
        
    except Exception as e:
        print(f"ERROR: Failed to add Q&A: {e}")
        raise

# -------- New functions for Questions Solved section --------
def get_solved_questions(limit: int = 50) -> List[Dict[str, Any]]:
    """Get all solved questions with edit/delete capability"""
    try:
        q = db.collection("solved_questions").where("status", "==", "active").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
        rows = []
        for doc in q.stream():
            d = doc.to_dict()
            ts = d.get("timestamp")
            d["id"] = doc.id
            d["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            rows.append(d)
        
        print(f"DEBUG: Found {len(rows)} solved questions")
        return rows
        
    except Exception as e:
        print(f"ERROR: Failed to get solved questions: {e}")
        return []

def update_solved_question(doc_id: str, question: str, answer: str) -> None:
    """Update a solved question"""
    try:
        # Get the old question first
        solved_doc = db.collection("solved_questions").document(doc_id).get()
        if not solved_doc.exists:
            raise ValueError("Question not found")
        
        old_data = solved_doc.to_dict()
        old_question = old_data.get("question", "")
        
        # Update solved_questions collection
        db.collection("solved_questions").document(doc_id).update({
            "question": question,
            "answer": answer,
            "updated_at": datetime.now()
        })
        
        # Update DOCTOR document - remove old answer and add new one
        doctor_ref = db.collection("DOCTOR").document("1")
        doc = doctor_ref.get()
        data = doc.to_dict() if doc.exists else {}
        ans_dict = data.get("ans", {}) or {}
        
        # Remove old answer if question changed
        if old_question in ans_dict and old_question != question:
            del ans_dict[old_question]
        
        # Add new answer
        ans_dict[question] = answer
        doctor_ref.set({"ans": ans_dict}, merge=True)
        
        # Update MongoDB
        try:
            from embedding import update_question_answer
            update_question_answer(old_question, question, answer)
            print(f"DEBUG: Successfully updated Q&A in MongoDB: {question}")
        except Exception as e:
            print(f"ERROR: Failed to update in MongoDB: {e}")
        
        print(f"DEBUG: Successfully updated solved question: {question}")
        
    except Exception as e:
        print(f"ERROR: Failed to update solved question: {e}")
        raise

def delete_solved_question(doc_id: str) -> None:
    """Delete a solved question"""
    try:
        # Get the question first
        solved_doc = db.collection("solved_questions").document(doc_id).get()
        if not solved_doc.exists:
            raise ValueError("Question not found")
        
        question_data = solved_doc.to_dict()
        question = question_data.get("question", "")
        
        # Mark as deleted in solved_questions collection
        db.collection("solved_questions").document(doc_id).update({
            "status": "deleted",
            "deleted_at": datetime.now()
        })
        
        # Remove from DOCTOR document
        doctor_ref = db.collection("DOCTOR").document("1")
        doc = doctor_ref.get()
        data = doc.to_dict() if doc.exists else {}
        ans_dict = data.get("ans", {}) or {}
        
        if question in ans_dict:
            del ans_dict[question]
            doctor_ref.set({"ans": ans_dict}, merge=True)
        
        # Remove from MongoDB
        try:
            from embedding import delete_question_answer
            delete_question_answer(question)
            print(f"DEBUG: Successfully deleted Q&A from MongoDB: {question}")
        except Exception as e:
            print(f"ERROR: Failed to delete from MongoDB: {e}")
        
        print(f"DEBUG: Successfully deleted solved question: {question}")
        
    except Exception as e:
        print(f"ERROR: Failed to delete solved question: {e}")
        raise