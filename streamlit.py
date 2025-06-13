import streamlit as st
from streamlit_authenticator import Authenticate
from streamlit_option_menu import option_menu
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np



# chargement du Dataframe traité 
df= pd.read_csv("https://raw.githubusercontent.com/iampdv/projet_2/main/df_final_2.csv",sep=",")
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w400"

#fonction de recherche

def recommander_films_similaires(df, titre_base, top_n=5):
    if titre_base not in df['title'].values:
        return f"Film '{titre_base}' non trouvé."

    # Combiner titre + genres pour créer une "signature textuelle"
    df['description'] = df['title'] + ' ' + df['genres_list'].astype(str)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])

    # Trouver l'index du film de base
    idx_base = df[df['title'] == titre_base].index[0]

    # Calcul des similarités cosinus
    cosine_sim = cosine_similarity(tfidf_matrix[idx_base], tfidf_matrix).flatten()

    # Récupère les indices des films les plus similaires (hors lui-même)
    similar_indices = cosine_sim.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx_base]  # exclure le film lui-même

    # Prendre les 10 films les plus similaires pour introduire de la variété
    top_candidates = similar_indices[:10]

    # Sélection aléatoire de top_n parmi les top 10
    random.shuffle(top_candidates)
    selected_indices = top_candidates[:top_n]

    return df.iloc[selected_indices][['title', 'year_release', 'genres_list', 'poster_path', 'vote_average', 'overview']].reset_index(drop=True)

# Nos données utilisateurs doivent respecter ce format
lesDonneesDesComptes = {
    'usernames': {
        'utilisateur': {
            'name': 'utilisateur',
            'password': 'utilisateurMDP', # En production, les mots de passe doivent être hachés
            'email': 'utilisateur@gmail.com',
            'failed_login_attemps': 0,
            'logged_in': False,
            'role': 'utilisateur'
        },
        'root': {
            'name': 'root',
            'password': 'rootMDP', # En production, les mots de passe doivent être hachés
            'email': 'admin@gmail.com',
            'failed_login_attemps': 0,
            'logged_in': False,
            'role': 'administrateur'
        },
        'Martin': {
            'name': 'Martin',
            'password': 'martin', # En production, les mots de passe doivent être hachés
            'email': 'admin@gmail.com',
            'failed_login_attemps': 0,
            'logged_in': False,
            'role': 'administrateur'
    }
} }

# Initialisation de l'authentificateur
authenticator = Authenticate(
    lesDonneesDesComptes,  # Les données des comptes
    "cookie_name",         # Le nom du cookie, un str quelconque
    "cookie_key",          # La clé du cookie, un str quelconque
    30,                    # Le nombre de jours avant que le cookie expire
)

# Tentative de connexion de l'utilisateur
authenticator.login()



# Définition de la fonction pour la page d'accueil
def accueil():
    st.markdown("<h1 style='text-align: center;'>🎥 Bienvenue sur MovieMatch 🎬</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Le compagnon parfait pour vos soirées cinéma</h3>", unsafe_allow_html=True)

    st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://external-preview.redd.it/tWkYFRTFlq1fwO9BmAdtXVIQas1Hljzsl3iyZFGSNhg.jpg?width=640&crop=smart&auto=webp&s=9abdf1392c1eb187b1182b47cf678c78382c59dc' width='400'>
    </div>
    """,
    unsafe_allow_html=True
)
    st.write("")
    st.header("🎞️ En panne d'inspiration pour un film ?")
    st.subheader("Laissez-vous guider par nos recommandations ou explorez les titres du moment.")

    st.markdown("""
<p style=''text-align: center;>
Tu cherches quoi regarder ce soir ? Entre deux popcorns, tape simplement le nom d’un film que tu aimes, et laisse la magie opérer.  
<br><br>
Grâce aux données de <strong>IMDb</strong> et <strong>TMDb</strong>, on te propose en quelques secondes trois films qui devraient te plaire autant, voire plus.  
Affiches, résumés, notes... tout est là pour t’aider à faire ton choix.  
<br><br>
Prépare ton plaid, choisis ton film, et profite du show ! 🍿✨
</p>
""", unsafe_allow_html=True)
    
#Définition de la fonction recommandation par film
def recommandations():
    st.title("Bienvenue sur la page de recommandation par titre de film:")

    film_utilisateur = st.text_input("Choisissez un nom de film (titre en anglais):")

    if film_utilisateur:
        titre_recherche_normalise = film_utilisateur.strip().lower()

        if 'title_normalise' not in df.columns:
            df['title_normalise'] = df['title'].astype(str).str.strip().str.lower()

        resultats_recherche = df[df['title_normalise'] == titre_recherche_normalise].copy()

        if not resultats_recherche.empty:
            st.success(f"Film(s) trouvé(s) pour '{film_utilisateur}' :")

            film_trouve = resultats_recherche.iloc[0]

            titre_film = film_trouve['title']
            poster_path = film_trouve.get('poster_path', None)
            vote_average = film_trouve['vote_average']
            genre_film = film_trouve["genres_list"]
            directeur_film = film_trouve["director_names"]
            overview_film = film_trouve["overview"]
            released_date = film_trouve["release_date"]
            casting = film_trouve["actor_names"]

            col1, col2 = st.columns(2)
            with col1:
                if pd.notna(poster_path):
                    full_poster_url = TMDB_IMAGE_BASE_URL + str(poster_path)
                    st.image(full_poster_url)
                else:
                    st.write(f"Pas de poster disponible pour : {titre_film}")
                    st.image("https://placehold.co/300x450?text=No+Poster", caption=titre_film)

            with col2:
                st.markdown(f"<p style='margin-bottom: 0;'>Titre :  <strong>{titre_film}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-bottom: 0;'>Note moyenne ⭐: <strong>{vote_average:.2f}/10</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-bottom: 0;'>Date de sortie 📅: <strong>{released_date}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-bottom: 0;'> Genres : <strong>{genre_film}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-bottom: 0;'> Directeur 📸: <strong>{directeur_film}</strong></p>", unsafe_allow_html=True)
                st.write(" ")
                st.write(overview_film)
                st.write(f"**Acteurs** : {casting}")

            st.subheader("")
            st.subheader(f'Nos recommandations avec votre film : {titre_film}')
            

            recommandations_df = recommander_films_similaires(df, titre_film, top_n=3)

            if isinstance(recommandations_df, str):
                st.warning(recommandations_df)
            elif not recommandations_df.empty:
                colonnes = st.columns(3)
                for i, (index, row) in enumerate(recommandations_df.iterrows()):
                    if i < 3:
                        with colonnes[i]:
                            #st.write(f"**Recommandation {i + 1} :**")
                            st.subheader(row.get('title', 'Titre inconnu'))
                            year_release = row.get('year_release')
                            st.write(f'Année de sortie **: {year_release}**')
                            poster_path_rec = row.get('poster_path', None)
                            if pd.notna(poster_path_rec):
                                image_url = TMDB_IMAGE_BASE_URL + str(poster_path_rec)
                                st.image(image_url, use_container_width=True)
                            else:
                                st.write("Aucune image disponible.")

                            rating = row.get('vote_average')
                            if rating is not None:
                                st.write(f"⭐ Note moyenne **: {rating}/10**")
                            else:
                                st.write("Note non disponible.")
                            overview = row.get('overview')
                            if overview is not None:
                                st.write(overview)
                            else:
                                st.write("Pas de synopsis disponible.")
        else:
            st.warning(f"Aucun film trouvé pour '{film_utilisateur}'")
# Définition de la fonction pour la page Base de donnée
def Base_de_donnees():
    st.markdown("<h1 style='text-align: center;'>Nos bases de données partenaire :</h1>", unsafe_allow_html=True)
    st.header("IMDB")
    st.image("data:image/png;base64,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")
    st.write("Lien [IMDB](https://www.imdb.com/fr/)")
    st.write('___') # ----- entre les deux 
    st.header("TMDB")
    st.image("data:image/png;base64,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")
    st.write("Lien [TMDB](https://www.themoviedb.org/?language=fr)")
# Définition de la fonction pour la recommandation par mot clé
def recommandation_mot_cle():
    st.title("Bienvenue sur la page de recommandation par mot clé :")
    st.write(" ")
    mot_recherche = st.text_input("Quel est votre mot clé :")

    if mot_recherche:  # S'assurer que l'utilisateur a saisi quelque chose
        mot_recherche_normalise = mot_recherche.strip().lower()

        # Préparer une colonne normalisée pour la recherche si elle n'existe pas déjà
        if 'title_normalise' not in df.columns:
            df['title_normalise'] = df['title'].astype(str).str.strip().str.lower()

        # Filtrer le DataFrame pour trouver les films correspondants
        mask = (
        df['description_complete'].astype(str).str.lower().str.contains(mot_recherche_normalise) |
        df['overview'].astype(str).str.lower().str.contains(mot_recherche_normalise) |
        df['title_normalise'].str.contains(mot_recherche_normalise)
)
        resultats_mot = df[mask]
        
        if not resultats_mot.empty:  # Vérifier si des résultats ont été trouvés
            st.success(f"Film(s) trouvé(s) pour '{mot_recherche}' :")

            # Si plusieurs films, on prend le premier pour l'affichage détaillé
            film_sortie = resultats_mot.iloc[0]

            # Extraire les valeurs individuelles
            titre_film = film_sortie['title']
            poster_path = film_sortie['poster_path']
            vote_average = film_sortie['vote_average']
            genre_film = film_sortie.get("genres_list", "Inconnu")
            directeur_film = film_sortie.get("director_names", "Inconnu")
            overview_film = film_sortie.get("overview", "Pas de résumé disponible.")
            released_date = film_sortie.get("release_date", "Date inconnue")
            casting = film_sortie.get("actor_names", "Inconnu")

            col1, col2 = st.columns(2)

            with col1:
                # Afficher l'image
                if pd.notna(poster_path):  # Vérifier si le chemin du poster n'est pas NaN
                    full_poster_url = TMDB_IMAGE_BASE_URL + str(poster_path)
                    st.image(full_poster_url)
                else:
                    st.write(f"Pas de poster disponible pour : {titre_film}")
                    st.image("https://placehold.co/300x450?text=No+Poster", caption=titre_film)

            with col2:
                # Afficher les informations comme du texte
                st.markdown(f"<p style='margin-bottom: 0;'>Titre :  <strong>{titre_film}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-bottom: 0;'>Vote rating ⭐: <strong>{vote_average:.2f}/10</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-bottom: 0;'>Released 📅: <strong>{released_date}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-bottom: 0;'>Genres : <strong>{genre_film}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-bottom: 0;'>Directors 📸: <strong>{directeur_film}</strong></p>", unsafe_allow_html=True)

                st.write(" ")
                st.write(overview_film)
                st.write(f"**Actors** : {casting}")

            autres_propositions = resultats_mot.iloc[1:]  # Exclure le premier film affiché

            if not autres_propositions.empty:
                st.title("")
                st.title(f'Propositions via votre mot clé : {mot_recherche}.')

                # Sélectionner jusqu'à 5 films aléatoires
            autres_films_random = autres_propositions.sample(n=min(5, len(autres_propositions)))

            for _, film in autres_films_random.iterrows():
                titre = film.get("title", "Titre inconnu")
                vote = film.get("vote_average")
                poster = film.get("poster_path")

                with st.container():
                    col1, col2 = st.columns(2)

                with col1:
                    if pd.notna(poster):
                        st.image(TMDB_IMAGE_BASE_URL + str(poster), width=200)
                    else:
                        st.image("https://placehold.co/100x150?text=No+Poster", width=100)

                with col2:
                    st.markdown(f"**{titre}**")
                    st.markdown(f"*Vote average ⭐ :* {vote}") 
# Définition de la fonction pour la page titre du moment
def titre_du_moment():
    st.title("Bienvenue sur la page des titres du moment  :")
    st.subheader(" Vous ne savez pas quoi regarder ? 🎥 ")
    st.write("Laissez-vous inspirez ⬇️:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        film_aleatoire1 = df.sample(1).iloc[0]
        titre_aleatoire_1 = film_aleatoire1["title"]
        overview_1 = film_aleatoire1["overview"]
        poster_path_aleatoire1 = film_aleatoire1["poster_path"]
        poster_tmdb = TMDB_IMAGE_BASE_URL + str(poster_path_aleatoire1)
        vote_average = film_aleatoire1["vote_average"]
        genre1= film_aleatoire1["genres_list"]
        st.image(poster_tmdb)
        st.write(f"⭐ {vote_average:.2f}/10")
        st.markdown( f"<p style='margin-bottom: 0;'><strong>{titre_aleatoire_1}</strong></p>",unsafe_allow_html=True)
        st.write("")
        st.write(overview_1)
        st.markdown(f"<strong>Genres du film :</strong><br>{genre1}", unsafe_allow_html=True)
    with col2:
        film_aleatoire2 = df.sample(1).iloc[0]
        titre_aleatoire_2 = film_aleatoire2["title"]
        overview_2 = film_aleatoire2["overview"]
        poster_path_aleatoire2 = film_aleatoire2["poster_path"]
        poster_tmdb2 = TMDB_IMAGE_BASE_URL + str(poster_path_aleatoire2)
        vote_average2 = film_aleatoire2["vote_average"]
        genre2= film_aleatoire2["genres_list"]
        st.image(poster_tmdb2)
        st.write(f"⭐ {vote_average2:.2f}/10")
        st.markdown( f"<p style='margin-bottom: 0;'> <strong>{titre_aleatoire_2}</strong></p>",unsafe_allow_html=True)
        st.write("")
        st.write(overview_2)
        st.markdown(f"<strong>Genres du film :</strong><br>{genre2}", unsafe_allow_html=True)
        
    with col3:
        film_aleatoire3 = df.sample(1).iloc[0]
        titre_aleatoire_3 = film_aleatoire3["title"]
        overview_3 = film_aleatoire3["overview"]
        poster_path_aleatoire3 = film_aleatoire3["poster_path"]
        poster_tmdb3 = TMDB_IMAGE_BASE_URL + str(poster_path_aleatoire3)
        vote_average3 = film_aleatoire3["vote_average"]
        genre3= film_aleatoire3["genres_list"]
        st.image(poster_tmdb3)
        st.write(f"⭐ {vote_average3:.2f}/10")
        st.markdown( f"<p style='margin-bottom: 0;'> <strong>{titre_aleatoire_3}</strong></p>",unsafe_allow_html=True)
        st.write("")
        st.write(overview_3)
        st.markdown(f"<strong>Genres du film :</strong><br>{genre3}", unsafe_allow_html=True)


#Logique de rendu conditionnel basée sur l'état d'authentification

if st.session_state["authentication_status"]:
    # L'utilisateur est authentifié
    MENU_KEY="main_menu"
    with st.sidebar:
        # Bouton de déconnexion dans la barre latérale
        authenticator.logout("Déconnexion", "main")
        st.write(f'Bienvenue *{st.session_state["name"]}*')

        # Menu de navigation dans la barre latérale
        selected = option_menu(
            menu_title=None,  # Pas de titre pour le menu
            options=["🏡 Accueil", "⭐ Titres du moment", "🎬 Recommandation par film","🔑 Recommandation par mot clé", "📚 Base de données"], # Options du menu
            menu_icon="cast", # Icône optionnelle pour le menu lui-même
            default_index=0, # Élément sélectionné par défaut
            key=MENU_KEY,
        )


    # Affichage du contenu de la page sélectionnée

    if selected == "🏡 Accueil":
        accueil()
    elif selected == "🎬 Recommandation par film":
        recommandations()
    elif selected == "⭐ Titres du moment":
        titre_du_moment()
    elif selected == "📚 Base de données":
        Base_de_donnees()
    elif selected == "🔑 Recommandation par mot clé":
        recommandation_mot_cle()
    

elif st.session_state["authentication_status"] is False:
    # L'authentification a échoué
    st.error("L'username ou le password est/sont incorrect")
elif st.session_state["authentication_status"] is None:
    # Aucune tentative de connexion ou champs vides
    st.warning('Les champs username et mot de passe doivent être remplis')


