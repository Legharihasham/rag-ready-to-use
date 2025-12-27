import google.generativeai as genai
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

class GeminiAPI:
    def __init__(self):
        """
        Initialize the Gemini API client
        
        Requires GOOGLE_API_KEY environment variable to be set in .env file
        """
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set in .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    def _is_contextual_match(self, question, context):
        """
        Intelligently determine if context is relevant to the question
        Uses semantic keyword matching and context analysis
        """
        if not context or len(context) == 0:
            return False
        
        # Extract key terms from question
        question_lower = question.lower()
        
        # Common question patterns that are likely answerable with university content
        university_terms = [
            'fee', 'admission', 'course', 'program', 'semester', 'credit', 'degree',
            'department', 'faculty', 'student', 'registration', 'enrollment', 'scholarship',
            'hostel', 'campus', 'library', 'exam', 'grade', 'cgpa', 'gpa', 'transcript',
            'deadline', 'requirement', 'eligibility', 'criteria', 'process', 'procedure',
            'form', 'application', 'document', 'certificate', 'convocation', 'graduation',
            'schedule', 'timetable', 'class', 'lecture', 'lab', 'project', 'thesis',
            'tuition', 'payment', 'challan', 'withdraw', 'drop', 'add', 'change',
            'professor', 'instructor', 'advisor', 'counselor', 'office', 'contact'
        ]
        
        # Check if question contains university-related terms
        has_university_terms = any(term in question_lower for term in university_terms)
        
        # Check context relevance scores
        high_relevance_chunks = [c for c in context if c.get("metadata", {}).get("relevance_score", 0) > 0.2]
        medium_relevance_chunks = [c for c in context if c.get("metadata", {}).get("relevance_score", 0) > 0.15]
        
        # More lenient matching logic
        if len(high_relevance_chunks) >= 1:
            return True
        if len(medium_relevance_chunks) >= 2 and has_university_terms:
            return True
        if len(context) >= 3 and has_university_terms:
            return True
            
        return False
    
    def generate_response(self, question, context, query_history=None):
        """
        Generate a response using Gemini model with given context
        
        Args:
            question: User's question
            context: Context from relevant PDF chunks
            query_history: Previous user queries for context
            
        Returns:
            Gemini's response
        """
        # Handle meta-queries about conversation history
        if query_history and len(query_history) > 0:
            lower_question = question.lower()
            meta_query_phrases = [
                "what did i ask before", "what was my previous question", 
                "what were my previous questions", "what did i ask previously", 
                "what have i asked", "what questions did i ask",
                "what was my last question", "what did i just ask",
                "previous query", "previous questions"
            ]
            
            if any(phrase in lower_question for phrase in meta_query_phrases):
                if len(query_history) == 1:
                    return f"Your previous question was: \"{query_history[0]}\""
                else:
                    response = "Here are your previous questions:\n\n"
                    for i, q in enumerate(reversed(query_history[-5:])):
                        response += f"{i+1}. \"{q}\"\n"
                    return response
        
        # Handle greetings and small talk
        greeting_response = self._handle_small_talk(question)
        if greeting_response:
            return greeting_response
        
        # Intelligent context relevance check
        if not self._is_contextual_match(question, context):
            return "I don't have enough relevant information to answer that question accurately. Please ask about topics related to university procedures, fees, courses, admissions, or other university-specific information."
        
        # Generate response with enhanced prompt
        prompt = self._create_prompt(question, context, query_history)
        
        try:
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
            
            generation_config = {
                "temperature": 0.3,  # Slightly higher for more natural responses
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            return self._post_process_response(response.text, question, context)
            
        except Exception as e:
            return f"I encountered an error while processing your question. Please try rephrasing or ask another question. Error: {str(e)}"
    
    def _handle_small_talk(self, question):
        """Handle greetings and common conversational patterns"""
        question_lower = question.lower().strip()
        
        greeting_phrases = ["hi", "hello", "hey", "greetings", "good morning", 
                           "good afternoon", "good evening", "how are you", 
                           "what's up", "nice to meet you", "how's it going", "howdy"]
        
        if any(question_lower == phrase or question_lower.startswith(phrase + " ") 
               for phrase in greeting_phrases):
            return "Hey! How's it going? What would you like to know about university matters?"
        
        small_talk_patterns = {
            "thanks": ["thank", "thanks", "thank you", "appreciate"],
            "goodbye": ["bye", "goodbye", "see you", "farewell", "good night"],
            "help": ["help me", "assist", "support", "guidance"],
            "capabilities": ["what can you do", "how can you help", "your capabilities"],
            "identity": ["who are you", "what are you", "your name"]
        }
        
        for category, phrases in small_talk_patterns.items():
            if any(phrase in question_lower for phrase in phrases):
                responses = {
                    "thanks": "You're welcome! Feel free to ask if you need anything else.",
                    "goodbye": "Goodbye! Come back anytime you have questions.",
                    "help": "I can help with university admissions, fees, courses, campus facilities, procedures, and more. What do you need?",
                    "capabilities": "I can answer questions about university admissions, fee structures, courses, campus life, procedures, and academic policies. How can I assist you?",
                    "identity": "I'm your university assistant chatbot, here to help with all your questions about university procedures, admissions, courses, and more. What would you like to know?"
                }
                return responses[category]
        
        return None
    
    def _post_process_response(self, response_text, question, context):
        """
        Post-process and validate the response
        Prevents unnecessary "I don't have information" responses
        """
        # Check if response is refusing to answer
        refusal_patterns = [
            "don't have enough information",
            "can't answer",
            "unable to answer",
            "not enough information",
            "insufficient information",
            "cannot provide"
        ]
        
        is_refusal = any(pattern in response_text.lower() for pattern in refusal_patterns)
        
        if is_refusal:
            # Double-check if we actually have good context
            has_good_context = any(
                chunk.get("metadata", {}).get("relevance_score", 0) > 0.2 
                for chunk in context
            )
            
            if has_good_context and len(context) >= 2:
                # Retry with a more assertive prompt
                retry_prompt = f"""
You are a university assistant. The user asked: "{question}"

You have relevant context available. Please answer the question directly using the information provided.
Do NOT say you don't have enough information. Extract and present what is available.

CONTEXT:
{chr(10).join([chunk["text"] for chunk in context[:5]])}

Provide a helpful answer based on this context:
"""
                try:
                    retry_response = self.model.generate_content(
                        retry_prompt,
                        generation_config={"temperature": 0.4, "max_output_tokens": 2048}
                    )
                    
                    # If retry also refuses, keep original
                    retry_text = retry_response.text
                    if not any(pattern in retry_text.lower() for pattern in refusal_patterns):
                        return retry_text
                except:
                    pass
        
        return response_text
    
    def generate_conversation_summary(self, query_history):
        """Generate a summary of conversation history for context"""
        if not query_history or len(query_history) <= 5:
            return None
            
        summary_prompt = f"""
Summarize the main topics from these user queries in 2-3 sentences. Focus on key information needs.

QUERIES:
{chr(10).join([f"- {q}" for q in query_history[-10:]])}

SUMMARY:
"""
        try:
            response = self.model.generate_content(summary_prompt)
            return response.text
        except:
            return None
    
    def _create_prompt(self, question, context, query_history=None):
        """Create an enhanced, industry-grade prompt"""
        
        system_prompt = """
You are an expert university assistant chatbot with deep knowledge of university procedures, policies, and operations. You communicate naturally and professionally, like a knowledgeable university advisor helping a student.

CORE PRINCIPLES:
1. **Accuracy First**: Provide precise, complete information from the context provided
2. **Always Extract Available Information**: If context contains ANY relevant information, use it—never say "I don't have enough information" unless context is truly empty or irrelevant
3. **Natural Communication**: Write like a human advisor, not a robot. Avoid phrases like "As an AI" or "According to the document"
4. **Clear Formatting**: Use bullet points, numbering, and spacing for readability
5. **Student-Friendly Language**: Explain clearly, avoid unnecessary jargon
6. **Never Cite Sources**: Don't mention document names or sources
7. **Partial Answers**: If you can answer part of a question, provide that information and note what's missing
8. **Context Awareness**: Use conversation history when relevant
9. **Comprehensive Responses**: Cover all aspects of the question thoroughly
10. **Practical Guidance**: Include actionable steps and helpful details

RESPONSE STRATEGY:
- Analyze the context carefully for ANY relevant information
- Extract all pertinent details related to the question
- Structure information logically and clearly
- Provide complete, helpful answers that students can act on
- If information is truly absent, be specific about what's missing

Remember: You're here to help students succeed. Be thorough, accurate, and supportive.
"""

        # Organize and format context intelligently
        sorted_context = sorted(
            context, 
            key=lambda x: x.get("metadata", {}).get("relevance_score", 0), 
            reverse=True
        )
        
        # Separate by source type for better organization
        pdf_chunks = [c for c in sorted_context if c["metadata"].get("type") == "pdf"]
        web_chunks = [c for c in sorted_context if c["metadata"].get("type") == "web"]
        
        context_sections = []
        
        if pdf_chunks:
            pdf_text = "\n\n".join([
                f"[Score: {chunk['metadata'].get('relevance_score', 0):.2f}]\n{chunk['text']}" 
                for chunk in pdf_chunks[:5]  # Top 5 most relevant
            ])
            context_sections.append(f"OFFICIAL DOCUMENTS:\n{pdf_text}")
            
        if web_chunks:
            web_text = "\n\n".join([
                f"[Score: {chunk['metadata'].get('relevance_score', 0):.2f}]\n{chunk['text']}" 
                for chunk in web_chunks[:5]
            ])
            context_sections.append(f"WEBSITE INFORMATION:\n{web_text}")
        
        combined_context = "\n\n---\n\n".join(context_sections)
        
        # Add conversation context
        conversation_context = ""
        if query_history and len(query_history) > 0:
            summary = self.generate_conversation_summary(query_history)
            if summary:
                conversation_context += f"\n\nCONVERSATION SUMMARY:\n{summary}\n"
            
            recent_queries = query_history[-5:]
            conversation_context += "\nRECENT QUESTIONS:\n" + "\n".join([f"• {q}" for q in recent_queries])
        
        # Construct final prompt with emphasis on using available information
        prompt = f"""{system_prompt}

AVAILABLE INFORMATION:
{combined_context}
{conversation_context}

STUDENT'S QUESTION:
{question}

INSTRUCTIONS:
Carefully review the available information above. Extract and present ALL relevant details that help answer the question. Provide a complete, well-structured response. Only state that information is unavailable if the context genuinely contains nothing related to the question.

YOUR RESPONSE:
"""
        
        return prompt