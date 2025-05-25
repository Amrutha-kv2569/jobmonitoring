import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import os

# ------------------------ Streamlit Page Config ------------------------
st.set_page_config(page_title="Smart Job Recommender", page_icon="🧠", layout="wide")

# ------------------------ Header ------------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Smart Job Recommender for Data Careers</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Scrape, Cluster, and Match Jobs Based on Your Skills</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.header("🔧 Job Search Controls")
    keyword = st.text_input("🔍 Job Keyword", "data science")
    pages = st.slider("📄 Pages to Scrape", 1, 5, 1)
    cluster_count = st.slider("🧩 Number of Clusters", 2, 10, 5)

    if st.button("🚀 Start Search & Clustering"):
        st.session_state['run_clustering'] = True

    st.markdown("---")
    user_skills = st.text_input("🎯 Your Skills", "python, machine learning, SQL")
    if st.button("🔍 Match My Skills"):
        st.session_state['run_matching'] = True


# ------------------------ Scraper ------------------------
@st.cache_data(show_spinner=False)
def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{}/all/India?search={}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page, keyword.replace(" ", "%20"))
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        job_blocks = soup.find_all("div", class_="ads-details")
        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Skills": skills,
                    "Summary": summary
                })
            except:
                continue

    return pd.DataFrame(jobs_list)


def cluster_jobs(df, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Skills'])

    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = model.fit_predict(X)

    with open("model.pkl", "wb") as f:
        pickle.dump((model, vectorizer), f)

    return df


# ------------------------ Job Cards ------------------------
def display_job_cards(df, title="Jobs", cluster=None):
    st.markdown(f"<h3 style='color: #6A5ACD;'>{title}</h3>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 2]:
            st.markdown(f"""
                <div style='background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid {"#6A5ACD" if cluster is None else f"#E91E63"};'>
                    <h4>{row['Title']}</h4>
                    <p><b>Company:</b> {row['Company']}<br>
                    <b>Location:</b> {row['Location']}<br>
                    <b>Experience:</b> {row['Experience']}<br>
                    <b>Skills:</b> {row['Skills']}</p>
                </div>
            """, unsafe_allow_html=True)

# ------------------------ Execution Triggers ------------------------

if 'run_clustering' in st.session_state and st.session_state['run_clustering']:
    with st.spinner("🔄 Scraping job listings..."):
        df = scrape_karkidi_jobs(keyword=keyword, pages=pages)

    if not df.empty:
        with st.spinner("⚙️ Clustering jobs by skills..."):
            df_clustered = cluster_jobs(df, n_clusters=cluster_count)

        st.success(f"✅ Scraped {len(df)} jobs and grouped them into {cluster_count} clusters.")
        for cluster_num in range(cluster_count):
            cluster_df = df_clustered[df_clustered['Cluster'] == cluster_num]
            if not cluster_df.empty:
                display_job_cards(cluster_df.head(4), title=f"🔗 Cluster {cluster_num + 1}", cluster=cluster_num)

        st.session_state['df_jobs'] = df_clustered
    else:
        st.warning("⚠️ No jobs found. Try changing the keyword or page range.")
    st.session_state['run_clustering'] = False

# ------------------------ Skill Matcher ------------------------
if 'run_matching' in st.session_state and st.session_state['run_matching'] and user_skills:
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model, vectorizer = pickle.load(f)

        user_vector = vectorizer.transform([user_skills])
        cluster_id = model.predict(user_vector)[0]

        st.info(f"🧲 Based on your skills, you're most aligned with **Cluster {cluster_id + 1}**.")

        if 'df_jobs' not in st.session_state:
            df_jobs = scrape_karkidi_jobs(keyword=keyword, pages=1)
            df_jobs['Cluster'] = model.predict(vectorizer.transform(df_jobs['Skills']))
        else:
            df_jobs = st.session_state['df_jobs']

        matched_jobs = df_jobs[df_jobs['Cluster'] == cluster_id]
        if not matched_jobs.empty:
            display_job_cards(matched_jobs.head(6), title="🎯 Jobs Matching Your Skills")
        else:
            st.warning("❌ No matching jobs found in this cluster.")
    else:
        st.error("❗ Please cluster jobs first before matching.")
    st.session_state['run_matching'] = False
