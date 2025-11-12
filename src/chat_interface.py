
import gradio as gr

def create_gradio(chat_fn, title="RAG Chatbot"):
    """Launch Gradio chat interface"""
    demo = gr.Interface(
        fn=chat_fn,
        inputs=gr.Textbox(lines=2, placeholder="Ask your question"),
        outputs=gr.Textarea(lines=25, max_lines=30),
        title=title
    )
    demo.launch(debug=True)
