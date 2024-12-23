# from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from langchain.docstore.document import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
class RecommentBook:
    def __init__(self, embedding_file, key, repo, csv_file, template):
        self.embedding = embedding_file
        self.key = key
        self.repo = repo
        self.csv_file = csv_file
        self.template = template
        self.data = pd.read_csv(self.csv_file)
        self.summary_title = dict(zip(self.data['summary'], self.data['title']))
        self.title_dict = dict(zip(self.data['title'], self.data['title']))  
        self.db = self.Documents()
        self.prompt = self.Prompt()
        self.llm = self.llms()
        self.rag = self.RAG(self.llm, self.prompt, self.db)

    def Documents(self):
        documents = [Document(page_content=row['summary']) for _, row in self.data.iterrows()]
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding)
        db = FAISS.from_documents(documents, embeddings)
        return db

    def Prompt(self):
        prompt = PromptTemplate(template=self.template, input_variables=['context', 'question'])
        return prompt

    def llms(self):
        llm = HuggingFaceHub(
            huggingfacehub_api_token=self.key,
            repo_id=self.repo,
            model_kwargs={'temperature': 1, 'max_length': 5000}
        )
        return llm

    def RAG(self, llm, prompt, db):
        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(),
            chain_type_kwargs={'prompt': prompt},
        )
        return rag

    # def find_title_summary(self, summary):
    #     query_document = Document(page_content=summary)
    #     result = self.db.similarity_search(query_document.page_content)
    #     not_query = result[0].page_content if result else None
    #     return self.summary_title.get(not_query, 'not found title')

    def find_title_summary(self, search_text):
            documents = [search_text] + self.data['summary'].tolist() + self.data['title'].tolist()
            vectorizer = TfidfVectorizer().fit_transform(documents)
            vectors = vectorizer.toarray()
            cosine_matrix = cosine_similarity(vectors)
            
            similar_summary_indices = cosine_matrix[0][1:len(self.data)+1]
            similar_title_indices = cosine_matrix[0][len(self.data)+1:]
            
            best_summary_index = np.argmax(similar_summary_indices)
            best_title_index = np.argmax(similar_title_indices)
            
            best_summary_score = similar_summary_indices[best_summary_index]
            best_title_score = similar_title_indices[best_title_index]
            
            threshold = 0.2
            
            if max(best_summary_score, best_title_score) > threshold:
                if best_summary_score > best_title_score:
                    matching_summary = self.data['summary'].iloc[best_summary_index]
                    return self.summary_title.get(matching_summary, 'not found title')
                else:
                    matching_title = self.data['title'].iloc[best_title_index]
                    return self.title_dict.get(matching_title, 'not found title')
                    
            return 'not found title'