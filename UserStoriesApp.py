# Import libraries 
import openai
import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# App main code 
def main():
    # Retrieve the key from the .env file
    openai.api_key = os.getenv("OPENAI_API_KEY", "The key was not found")

    # Load the UI 
    st.set_page_config(page_title="User Stories Generator App")
    st.header("Analyse you user's feedback 💬")
    fb_doc = st.file_uploader("Upload the user's feedback", type=".txt")
    ctx_doc = st.file_uploader("Upload role and context", type=".txt")

    # Read the document
    if fb_doc is not None:
        # Extract text from the document
        feedback = fb_doc.read().decode("utf-8")

        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(feedback)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Openai request prompt
        prompt = f"""Could you write user stories using the framework: 'As a [user role], 
            I would like to [action], so that [benefit],' based on the previous feedback from 
            the user found in the document? Please separate them with "\n"
            """
        docs = knowledge_base.similarity_search(prompt)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=prompt)

        # Get ai responses
        # user_stories = response.choices[0].message.content
        user_stories_list = response.split('\n')
        user_stories_list = [string.strip() for string in user_stories_list]
        print(user_stories_list)
        

        # Print user stories from ai response
        st.subheader("User stories:")
        st.write(response)

if __name__ == '__main__':
    main()