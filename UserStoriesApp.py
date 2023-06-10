# Import libraries 
import openai
import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

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
    if fb_doc and ctx_doc is not None:
        # Extract text from the documents
        feedback = fb_doc.read().decode("utf-8")
        context = ctx_doc.read().decode("utf-8")

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
        prompt = f"""{context}.Could you write user stories using the framework: 'As a [user role], 
            I would like to [action], so that [benefit],' based on the previous feedback from 
            the user found in the document? End paragraph after each user story.
            """
        docs = knowledge_base.similarity_search(prompt)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=prompt)

        # Get ai responses
        # user_stories = response.choices[0].message.content
        user_stories_list = response.split('\n\n')
        user_stories_list = [string.strip() for string in user_stories_list]
        print(user_stories_list)
        # Display user stories from ai response
        st.subheader("User stories:")
        
        # For each stories ask OpenAI for definitions of done
        for index, story in enumerate(user_stories_list):
            st.write(f"User story {index + 1}: {story}")
            prompt = f"""{context}. Please write a set of user criteria based on the following user stoy: {story}
            """
            # Generate user definitions of done
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            #=10,  # Generate 3 user stories
            stop=None,
            )
            def_done = response.choices[0].message.content

            st.write(def_done)

if __name__ == '__main__':
    main()