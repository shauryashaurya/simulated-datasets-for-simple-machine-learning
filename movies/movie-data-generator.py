import random
import pandas as pd
import numpy as np
from faker import Faker
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, date
import json
import requests
import io
import zipfile
from pathlib import Path
import os
import csv
import re

"""
Generating movies based on the simple ontology:
 
    Class: Movie                                                         
    Properties (data):                                                         
        - title : string                                                         
        - releaseYear : integer                                                         
        - duration : integer                                                         
        - rating : float     
        - plot: string
        - budget: integer
        - revenue: integer        
    Properties (object):                                                         
        - hasActor -> Actor                                                         
        - hasDirector -> Director                                                         
        - belongsToGenre -> Genre                                                         

    Class: Person                                                         
    Properties (data):                                                         
        - name : string                                                         
        - birthDate : date 
        - popularity
        - movie_ids

    Class: Actor (subclassOf Person)    
    Properties (data):                                                         
        - character_ids	
    Properties (object):                                                         
        - playsCharacter -> Character                                                         

    Class: Director (subclassOf Person)                                                         
    (no special object property here, but reuses name/birthDate)                                                         

    Class: Character                                                         
    Properties (data):                                                         
        - name : string                                                         

    Class: Genre                                                         
    Properties (data):                                                         
        - name : string


        
Clear citations for all the data sources that are used or referenced in the code:

### Primary Data Sources Used in the Code

1. **MovieLens Dataset**
   - Source: GroupLens Research
   - URL: https://grouplens.org/datasets/movielens/
   - Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872
   - License: Available for non-commercial use under creative commons license

2. **Film Directors List**
   - Source: GitHub Gist by drmonkeysee
   - URL: https://gist.githubusercontent.com/drmonkeysee/9211094/raw/f5a5c57a696fbd56e2c83ce971e2a51e4c701bd7/film_directors.txt
   - This is a community-created list and should be used as sample data only

3. **Actors List**
   - Source: GitHub Gist by mbejda
   - URL: https://gist.githubusercontent.com/mbejda/1e9c9e4e18234bc6ff7e/raw/a453e913a4dcd5829ad2b4f777ddc400f696554f/Actors
   - This is a community-created list and should be used as sample data only
"""



# Initialize faker
fake = Faker()

# Add a seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake.seed_instance(RANDOM_SEED)

# Configuration class to set the number of entities to generate
@dataclass
class GeneratorConfig:
    num_genres: int = 20
    num_directors: int = 100
    num_actors: int = 500
    num_characters: int = 1000
    num_movies: int = 1000
    min_actors_per_movie: int = 3
    max_actors_per_movie: int = 15
    min_genres_per_movie: int = 1
    max_genres_per_movie: int = 3
    
    # Path for downloading/loading data
    data_path: str = "movie-data"
    
    # Option to use real data (if False, all data will be generated with Faker)
    use_real_data: bool = True


# Data classes representing the ontology
@dataclass
class Genre:
    id: int
    name: str


@dataclass
class Person:
    id: int
    name: str
    birth_date: datetime
    # Additional fields for realism
    popularity: float = 0.0


@dataclass
class Director(Person):
    # Movies directed (will be populated later)
    movie_ids: List[int] = field(default_factory=list)


@dataclass
class Character:
    id: int
    name: str
    # Additional fields for realism
    description: str = ""
    movie_id: int = -1


@dataclass
class Actor(Person):
    character_ids: List[int] = field(default_factory=list)
    # Additional fields for realism
    movie_ids: List[int] = field(default_factory=list)
    
    def plays_character(self, character_id: int):
        if character_id not in self.character_ids:
            self.character_ids.append(character_id)
    
    def appears_in(self, movie_id: int):
        if movie_id not in self.movie_ids:
            self.movie_ids.append(movie_id)


@dataclass
class Movie:
    id: int
    title: str
    release_year: int
    duration: int  # in minutes
    rating: float
    director_id: int
    actor_ids: List[int] = field(default_factory=list)
    genre_ids: List[int] = field(default_factory=list)
    # Additional fields for realism
    plot: str = ""
    budget: int = 0
    revenue: int = 0
    
    def has_actor(self, actor_id: int):
        if actor_id not in self.actor_ids:
            self.actor_ids.append(actor_id)
    
    def has_director(self, director_id: int):
        self.director_id = director_id
        
    def belongs_to_genre(self, genre_id: int):
        if genre_id not in self.genre_ids:
            self.genre_ids.append(genre_id)


class MovieDataGenerator:
    def __init__(self, config: GeneratorConfig = None):
        self.config = config or GeneratorConfig()
        self.genres: Dict[int, Genre] = {}
        self.directors: Dict[int, Director] = {}
        self.actors: Dict[int, Actor] = {}
        self.characters: Dict[int, Character] = {}
        self.movies: Dict[int, Movie] = {}
        
        # MovieLens data storage
        self.ml_movies = None
        self.ml_ratings = None
        self.ml_links = None
        self.ml_tags = None
        
        # Real director and actor names from datasets
        self.real_directors = []
        self.real_actors = []
        self.real_characters = []
        
        # Initialize the data directory
        Path(self.config.data_path).mkdir(exist_ok=True)
        
        # Load or generate real-world movie data
        if self.config.use_real_data:
            self._load_movielens_data()
            self._load_real_names_data()
    
    def _load_movielens_data(self):
        """Load MovieLens small dataset for realistic movie titles and metadata"""
        # Create data directory if it doesn't exist
        Path(self.config.data_path).mkdir(exist_ok=True)
        
        # URLs for MovieLens small dataset
        movielens_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        
        # Local file paths
        zip_path = os.path.join(self.config.data_path, "ml-latest-small.zip")
        
        # Download the data if it doesn't exist
        if not os.path.exists(zip_path):
            print(f"Downloading MovieLens small dataset from {movielens_url}")
            r = requests.get(movielens_url)
            with open(zip_path, 'wb') as f:
                f.write(r.content)
                
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.data_path)
        
        # Load the data
        data_dir = os.path.join(self.config.data_path, "ml-latest-small")
        self.ml_movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
        self.ml_ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
        self.ml_links = pd.read_csv(os.path.join(data_dir, "links.csv"))
        self.ml_tags = pd.read_csv(os.path.join(data_dir, "tags.csv"))
        
        # Perform some initial data cleaning
        # Extract year from titles, e.g., "Toy Story (1995)" -> "Toy Story", 1995
        self.ml_movies['original_title'] = self.ml_movies['title']
        self.ml_movies['extracted_year'] = self.ml_movies['title'].str.extract(r'\((\d{4})\)$').astype('float')
        self.ml_movies['title'] = self.ml_movies['title'].str.replace(r' \(\d{4}\)$', '', regex=True)
        
        print(f"Loaded MovieLens data: {len(self.ml_movies)} movies")
    
    def _load_real_names_data(self):
        """
        Load real actor, director, and character names from freely available sources
        """
        # Download some real actor/director names from GitHub gists
        directors_url = "https://gist.githubusercontent.com/drmonkeysee/9211094/raw/f5a5c57a696fbd56e2c83ce971e2a51e4c701bd7/film_directors.txt"
        actors_url = "https://gist.githubusercontent.com/mbejda/1e9c9e4e18234bc6ff7e/raw/a453e913a4dcd5829ad2b4f777ddc400f696554f/Actors"
        
        try:
            # Get directors
            r = requests.get(directors_url)
            if r.status_code == 200:
                directors = r.text.strip().split('\n')
                self.real_directors = [name for name in directors if name.strip()]
                print(f"Loaded {len(self.real_directors)} real director names")
            
            # Get actors
            r = requests.get(actors_url)
            if r.status_code == 200:
                actors = r.text.strip().split('\n')
                self.real_actors = [name for name in actors if name.strip()]
                print(f"Loaded {len(self.real_actors)} real actor names")
        except Exception as e:
            print(f"Warning: Could not load real names data: {e}")
            print("Will use Faker to generate names instead")
        
        # Also create a list of common/famous character names from movies
        self.real_characters = [
            "James Bond", "Indiana Jones", "Luke Skywalker", "Darth Vader", "Han Solo",
            "Princess Leia", "Rocky Balboa", "John McClane", "Ellen Ripley", "Sarah Connor",
            "Jack Sparrow", "Forrest Gump", "Hannibal Lecter", "Norman Bates", "Michael Corleone",
            "Tony Montana", "Travis Bickle", "Tyler Durden", "The Joker", "Batman",
            "Superman", "Wonder Woman", "Iron Man", "Captain America", "Thor",
            "Hulk", "Black Widow", "Dorothy Gale", "Scarlett O'Hara", "Rick Blaine",
            "Holly Golightly", "Tony Stark", "Vito Corleone", "Atticus Finch", "Clarice Starling",
            "Maximus", "Neo", "Morpheus", "Trinity", "Gandalf",
            "Frodo Baggins", "Aragorn", "Legolas", "Gollum", "Harry Potter",
            "Hermione Granger", "Ron Weasley", "Severus Snape", "Albus Dumbledore", "Voldemort",
            "Katniss Everdeen", "Peeta Mellark", "Jack Dawson", "Rose DeWitt Bukater", "John Rambo",
            "John Wick", "Ethan Hunt", "Jason Bourne", "Marty McFly", "Doc Brown",
            "Ferris Bueller", "Freddy Krueger", "Michael Myers", "Laurie Strode", "Jason Voorhees",
            "Godzilla", "King Kong", "E.T.", "The Terminator", "John Connor",
            "Wolverine", "Professor X", "Magneto", "Storm", "Cyclops",
            "Black Panther", "Doctor Strange", "Captain Marvel", "Thanos", "Loki",
            "Sherlock Holmes", "Doctor Watson", "Agent Smith", "Gollum", "Sauron"
        ]
        
        # Add common character first names and descriptive titles
        character_first_names = [
            "Captain", "Doctor", "Professor", "Agent", "Sheriff", "Detective", "Officer",
            "General", "Colonel", "Sergeant", "King", "Queen", "Prince", "Princess",
            "Chief", "Mr.", "Mrs.", "Ms.", "Sir", "Lady", "Lord", "Duke", "Duchess"
        ]
        
        # Add some common last names for characters
        character_last_names = [
            "Smith", "Jones", "Brown", "Johnson", "Williams", "Davis", "Miller", "Wilson",
            "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Clark",
            "Lewis", "Walker", "Hall", "Young", "Allen", "Wright", "Scott", "Green", "Baker"
        ]
        
        # Create more character names by combining elements
        for _ in range(100):
            if random.random() < 0.7:  # 70% of the time use a full name
                self.real_characters.append(f"{random.choice(character_first_names)} {random.choice(character_last_names)}")
            else:  # 30% of the time use just a title and last name
                self.real_characters.append(f"{random.choice(character_first_names)} {fake.last_name()}")
        
        # Deduplicate
        self.real_characters = list(set(self.real_characters))
        print(f"Prepared {len(self.real_characters)} character names")
    
    def _extract_genres_from_movielens(self):
        """Extract unique genres from MovieLens dataset"""
        all_genres = set()
        for genres_str in self.ml_movies['genres']:
            genres = genres_str.split('|')
            for genre in genres:
                if genre != "(no genres listed)":
                    all_genres.add(genre)
        return list(all_genres)
    
    def generate_genres(self):
        """Generate genre data"""
        # If using real data, extract genres from MovieLens
        if self.config.use_real_data and self.ml_movies is not None:
            genre_names = self._extract_genres_from_movielens()
            # If we need more genres than what's in MovieLens, generate additional ones
            additional_genres = max(0, self.config.num_genres - len(genre_names))
            for i in range(additional_genres):
                genre_names.append(fake.word().capitalize())
        else:
            # Generate fake genre names
            genre_names = [fake.word().capitalize() for _ in range(self.config.num_genres)]
        
        # Create Genre objects
        for i, name in enumerate(genre_names[:self.config.num_genres]):
            self.genres[i] = Genre(id=i, name=name)
        
        print(f"Generated {len(self.genres)} genres")
        return self.genres
    
    def _get_realistic_birth_date(self, role_type='actor', min_age=18, max_age=80):
        """Generate a realistic birth date based on role type"""
        # Directors tend to be older than actors on average
        if role_type == 'director':
            min_age = max(30, min_age)  # Directors are usually at least 30
            max_age = min(90, max_age)  # Directors can be quite old
        
        # Generate a birth date
        return fake.date_of_birth(minimum_age=min_age, maximum_age=max_age)
    
    def generate_directors(self):
        """Generate director data with real names where possible"""
        # Use real director names if available
        real_names = []
        if self.config.use_real_data and self.real_directors:
            # Use names from our pre-loaded real directors
            real_names = self.real_directors.copy()
            random.shuffle(real_names)
        
        # Generate directors
        for i in range(self.config.num_directors):
            # Try to use a real name if available
            if i < len(real_names):
                name = real_names[i]
            else:
                # Fall back to Faker
                name = fake.name()
            
            # Generate a realistic birth date
            birth_date = self._get_realistic_birth_date(role_type='director')
            
            # Create Director object
            self.directors[i] = Director(
                id=i,
                name=name,
                birth_date=birth_date,
                movie_ids=[],  # Will be populated later
                popularity=random.uniform(0.0, 10.0)  # Random popularity score
            )
        
        print(f"Generated {len(self.directors)} directors")
        return self.directors
    
    def generate_characters(self):
        """Generate character data with known character names where possible"""
        # Use real character names if available
        real_names = []
        if self.config.use_real_data and self.real_characters:
            real_names = self.real_characters.copy()
            random.shuffle(real_names)
        
        # Generate a pool of unique character descriptions
        descriptions = [
            "The protagonist",
            "The antagonist",
            "A supporting character",
            "The love interest",
            "The mentor",
            "The sidekick",
            "The comic relief",
            "The villain",
            "The anti-hero",
            "The wise elder",
            "The rookie",
            "The mysterious stranger",
            "The loyal friend",
            "The betrayer",
            "The reluctant hero"
        ]
        
        # Generate characters
        for i in range(self.config.num_characters):
            # Try to use a real character name if available
            if i < len(real_names):
                name = real_names[i]
            else:
                # Fall back to generated names
                name_type = random.randint(0, 3)
                if name_type == 0:
                    # Full name
                    name = fake.name()
                elif name_type == 1:
                    # First name only
                    name = fake.first_name()
                elif name_type == 2:
                    # Name with title
                    titles = ["Dr.", "Professor", "Captain", "Agent", "Detective", "Officer", "Major", "Senator"]
                    name = f"{random.choice(titles)} {fake.last_name()}"
                else:
                    # Fantasy/unusual names
                    syllables = ['ba', 'be', 'bi', 'bo', 'bu', 'da', 'de', 'di', 'do', 'du', 
                                'fa', 'fe', 'fi', 'fo', 'fu', 'ga', 'ge', 'gi', 'go', 'gu', 
                                'ha', 'he', 'hi', 'ho', 'hu', 'ja', 'je', 'ji', 'jo', 'ju', 
                                'ka', 'ke', 'ki', 'ko', 'ku', 'la', 'le', 'li', 'lo', 'lu', 
                                'ma', 'me', 'mi', 'mo', 'mu', 'na', 'ne', 'ni', 'no', 'nu', 
                                'pa', 'pe', 'pi', 'po', 'pu', 'ra', 're', 'ri', 'ro', 'ru', 
                                'sa', 'se', 'si', 'so', 'su', 'ta', 'te', 'ti', 'to', 'tu', 
                                'va', 've', 'vi', 'vo', 'vu', 'za', 'ze', 'zi', 'zo', 'zu']
                    name = ''.join(random.choice(syllables) for _ in range(random.randint(2, 4)))
                    name = name.capitalize()
            
            # Generate a description
            description = random.choice(descriptions)
            
            # Create Character object
            self.characters[i] = Character(
                id=i,
                name=name,
                description=description,
                movie_id=-1  # Will be assigned later
            )
        
        print(f"Generated {len(self.characters)} characters")
        return self.characters
    
    def generate_actors(self):
        """Generate actor data with real names where possible"""
        # Use real actor names if available
        real_names = []
        if self.config.use_real_data and self.real_actors:
            real_names = self.real_actors.copy()
            random.shuffle(real_names)
        
        # Generate actors
        for i in range(self.config.num_actors):
            # Try to use a real name if available
            if i < len(real_names):
                name = real_names[i]
                popularity = random.uniform(0.0, 10.0)
            else:
                # Fall back to Faker
                name = fake.name()
                popularity = random.uniform(0.0, 10.0)
            
            # Generate a realistic birth date
            birth_date = self._get_realistic_birth_date(role_type='actor')
            
            # Create Actor object
            self.actors[i] = Actor(
                id=i,
                name=name,
                birth_date=birth_date,
                character_ids=[],  # Will be populated later
                movie_ids=[],      # Will be populated later
                popularity=popularity
            )
        
        print(f"Generated {len(self.actors)} actors")
        return self.actors
    
    def generate_movies(self):
        """Generate movie data with real movie details where possible"""
        # Get the current year for calculating release years
        current_year = datetime.now().year
        
        # If using real data and MovieLens is available, use its data for movie titles and metadata
        if self.config.use_real_data and self.ml_movies is not None:
            # Sample movies from MovieLens (up to the number we need)
            num_ml_movies = min(self.config.num_movies, len(self.ml_movies))
            sampled_movies = self.ml_movies.sample(
                n=num_ml_movies,
                random_state=RANDOM_SEED
            ).reset_index(drop=True)
            
            # Create mapping from MovieLens genre to our genre IDs
            ml_genre_map = {}
            for genre_id, genre in self.genres.items():
                ml_genre_map[genre.name] = genre_id
            
            # Process real movies
            for i in range(num_ml_movies):
                ml_row = sampled_movies.iloc[i]
                ml_movie_id = ml_row['movieId']
                
                # Extract movie data from MovieLens
                title = ml_row['title']
                
                # Extract release year from MovieLens
                if pd.notna(ml_row['extracted_year']):
                    release_year = int(ml_row['extracted_year'])
                else:
                    release_year = random.randint(1950, current_year - 1)
                
                # Generate a realistic duration
                # Use a normal distribution centered around typical movie lengths
                # Most movies are between 90-150 minutes
                duration = int(max(60, min(240, np.random.normal(120, 20))))
                
                # Get rating from MovieLens or generate
                movie_ratings = self.ml_ratings[self.ml_ratings['movieId'] == ml_movie_id]
                if not movie_ratings.empty:
                    rating = round(movie_ratings['rating'].mean(), 1)
                else:
                    # Generate a bell-curved rating between 1.0 and 5.0
                    rating = min(5.0, max(1.0, np.random.normal(3.5, 0.8)))
                    rating = round(rating, 1)
                
                # Generate plot
                plot = fake.paragraph()
                
                # Generate budget and revenue
                # Budgets typically range from $1M to $200M
                budget = int(max(1e6, min(2e8, np.random.lognormal(17, 1))))
                # Revenues can be 0 (flop) to several times the budget (hit)
                revenue_factor = max(0, np.random.normal(2, 1.5))  # Mean of 2x budget
                revenue = int(budget * revenue_factor)
                
                # Create Movie object
                movie = Movie(
                    id=i,
                    title=title,
                    release_year=release_year,
                    duration=duration,
                    rating=rating,
                    director_id=random.choice(list(self.directors.keys())),
                    actor_ids=[],
                    genre_ids=[],
                    plot=plot,
                    budget=budget,
                    revenue=revenue
                )
                
                # Add actors to the movie
                num_actors = random.randint(
                    self.config.min_actors_per_movie,
                    self.config.max_actors_per_movie
                )
                actor_ids = random.sample(list(self.actors.keys()), num_actors)
                for actor_id in actor_ids:
                    movie.has_actor(actor_id)
                    self.actors[actor_id].appears_in(i)
                
                # Add the director's movie to their filmography
                self.directors[movie.director_id].movie_ids.append(i)
                
                # Extract genres from MovieLens and map to our genre IDs
                ml_genres = ml_row['genres'].split('|')
                for genre_name in ml_genres:
                    if genre_name in ml_genre_map and genre_name != "(no genres listed)":
                        movie.belongs_to_genre(ml_genre_map[genre_name])
                
                # If no genres were mapped, assign random ones
                if not movie.genre_ids:
                    num_genres = random.randint(
                        self.config.min_genres_per_movie,
                        self.config.max_genres_per_movie
                    )
                    genre_ids = random.sample(list(self.genres.keys()), num_genres)
                    for genre_id in genre_ids:
                        movie.belongs_to_genre(genre_id)
                
                self.movies[i] = movie
            
            # Generate additional movies if needed
            if self.config.num_movies > num_ml_movies:
                self._generate_synthetic_movies(start_idx=num_ml_movies)
        else:
            # Generate entirely synthetic movie data
            self._generate_synthetic_movies(start_idx=0)
        
        print(f"Generated {len(self.movies)} movies")
        return self.movies
    
    def _generate_synthetic_movies(self, start_idx=0):
        """Generate synthetic movie data for when we need more than real data provides"""
        current_year = datetime.now().year
        
        # Movie title generation components
        movie_adjectives = [
            "Lost", "Hidden", "Secret", "Last", "Final", "First", "Eternal", "Dark",
            "Golden", "Silver", "Red", "Blue", "Green", "Black", "White", "Bloody",
            "Broken", "Shattered", "Rising", "Falling", "Burning", "Frozen", "Endless",
            "Silent", "Whispered", "Screaming", "Laughing", "Dancing", "Running",
            "Forgotten", "Forbidden", "Sacred", "Cursed", "Blessed", "Ancient", "Modern",
            "Quantum", "Cosmic", "Digital", "Virtual", "Cyber", "Mystic", "Magical"
        ]
        
        movie_nouns = [
            "Knight", "Warrior", "King", "Queen", "Princess", "Prince", "Hero", "Legend",
            "Empire", "Kingdom", "City", "World", "Planet", "Galaxy", "Universe", "Dimension",
            "Dream", "Nightmare", "Fantasy", "Reality", "Memory", "Future", "Past", "Present",
            "Heart", "Soul", "Mind", "Body", "Spirit", "Shadow", "Light", "Darkness",
            "Mountains", "Ocean", "Sky", "Earth", "Fire", "Water", "Wind", "Storm",
            "Star", "Moon", "Sun", "Comet", "Asteroid", "Pulsar", "Quasar", "Nebula",
            "Machine", "Robot", "Android", "Cyborg", "Computer", "Program", "Code", "Algorithm"
        ]
        
        movie_formats = [
            "{adj} {noun}",
            "The {adj} {noun}",
            "{noun} of {adj}",
            "{adj} {noun}s",
            "The {noun}",
            "{noun}",
            "{adj} {noun}: {adj} {noun}",
            "{noun}: {adj} {noun}",
            "The {noun} of the {noun}",
            "{noun} & {noun}",
            "{adj} {noun} {number}",
            "{noun} {number}",
            "{adj} {adj} {noun}"
        ]
        
        for i in range(start_idx, self.config.num_movies):
            # Generate a title
            format_str = random.choice(movie_formats)
            title = format_str.format(
                adj=random.choice(movie_adjectives),
                noun=random.choice(movie_nouns),
                number=random.randint(2, 10)
            )
            
            # Generate release year (weighted towards more recent years)
            years_ago = int(np.random.exponential(15))
            release_year = min(current_year - 1, current_year - years_ago)
            release_year = max(1950, release_year)  # Don't go earlier than 1950
            
            # Generate a rating (weighted bell curve)
            rating = min(5.0, max(1.0, np.random.normal(3.5, 0.8)))
            rating = round(rating, 1)
            
            # Generate a duration (normal distribution around 120 minutes)
            duration = int(max(60, min(240, np.random.normal(120, 20))))
            
            # Generate budget and revenue
            budget = int(max(1e6, min(2e8, np.random.lognormal(17, 1))))
            revenue_factor = max(0, np.random.normal(2, 1.5))  # Mean of 2x budget
            revenue = int(budget * revenue_factor)
            
            # Create Movie object
            movie = Movie(
                id=i,
                title=title,
                release_year=release_year,
                duration=duration,
                rating=rating,
                director_id=random.choice(list(self.directors.keys())),
                actor_ids=[],
                genre_ids=[],
                plot=fake.paragraph(),
                budget=budget,
                revenue=revenue
            )
            
            # Add actors to the movie
            num_actors = random.randint(
                self.config.min_actors_per_movie,
                self.config.max_actors_per_movie
            )
            actor_ids = random.sample(list(self.actors.keys()), num_actors)
            for actor_id in actor_ids:
                movie.has_actor(actor_id)
                self.actors[actor_id].appears_in(i)
            
            # Add the director's movie to their filmography
            self.directors[movie.director_id].movie_ids.append(i)
            
            # Add genres to the movie
            num_genres = random.randint(
                self.config.min_genres_per_movie,
                self.config.max_genres_per_movie
            )
            genre_ids = random.sample(list(self.genres.keys()), num_genres)
            for genre_id in genre_ids:
                movie.belongs_to_genre(genre_id)
            
            self.movies[i] = movie
    
    def assign_characters_to_actors(self):
        """Assign characters to actors based on the movies they're in"""
        # Dictionary to track which characters are assigned to which actors in which movies
        movie_character_assignments = {}
        
        for movie_id, movie in self.movies.items():
            # Create a pool of characters for this movie
            num_characters_needed = len(movie.actor_ids)
            
            # Get characters that haven't been used too much
            available_characters = [
                char_id for char_id, char in self.characters.items()
                if char_id not in movie_character_assignments and char.movie_id == -1
            ]
            
            # If we need more characters than are available, reuse some
            if len(available_characters) < num_characters_needed:
                additional_needed = num_characters_needed - len(available_characters)
                # Get characters that have been used the least
                character_usage = {char_id: 0 for char_id in self.characters.keys()}
                for movie_chars in movie_character_assignments.values():
                    for char_id in movie_chars.values():
                        character_usage[char_id] = character_usage.get(char_id, 0) + 1
                
                # Sort by usage and add the least used ones
                sorted_chars = sorted(character_usage.items(), key=lambda x: x[1])
                additional_chars = [char_id for char_id, _ in sorted_chars[:additional_needed]]
                available_characters.extend(additional_chars)
            
            # Randomly select characters for this movie
            movie_characters = random.sample(available_characters, num_characters_needed)
            
            # Assign characters to actors in this movie
            movie_char_assignments = {}
            for actor_id, char_id in zip(movie.actor_ids, movie_characters):
                movie_char_assignments[actor_id] = char_id
                self.actors[actor_id].plays_character(char_id)
                # Update character's movie_id if it wasn't assigned yet
                if self.characters[char_id].movie_id == -1:
                    self.characters[char_id].movie_id = movie_id
            
            movie_character_assignments[movie_id] = movie_char_assignments
        
        print(f"Assigned characters to actors across {len(movie_character_assignments)} movies")
        return movie_character_assignments
    
    def generate_all_data(self):
        """Generate all data based on the configuration"""
        self.generate_genres()
        self.generate_directors()
        self.generate_characters()
        self.generate_actors()
        self.generate_movies()
        self.assign_characters_to_actors()
        
        return {
            "genres": self.genres,
            "directors": self.directors,
            "characters": self.characters,
            "actors": self.actors,
            "movies": self.movies
        }
    
    def _safe_csv_string(self, value):
        """Format a value for CSV, escaping special characters"""
        if isinstance(value, str):
            # Replace double quotes with two double quotes (CSV escaping)
            value = value.replace('"', '""')
            # If the value contains commas, quotes, or newlines, wrap in quotes
            if ',' in value or '"' in value or '\n' in value:
                value = f'"{value}"'
        elif isinstance(value, list):
            # For lists, join with a semicolon (to avoid CSV delimiter issues)
            value = ';'.join(str(item) for item in value)
        elif isinstance(value, datetime):
            # Format dates as ISO string
            value = value.strftime('%Y-%m-%d')
        elif value is None:
            value = ""
        return str(value)
    
    def export_to_json(self, output_dir="output"):
        """Export the generated data to JSON files"""
        Path(output_dir).mkdir(exist_ok=True)
        
        def obj_to_dict(obj):
            """Convert dataclass instances to dictionaries"""
            if hasattr(obj, '__dict__'):
                result = obj.__dict__.copy()
                # Convert nested datetime objects to strings
                for key, value in result.items():
                    if isinstance(value, datetime) or isinstance(value, date):
                        result[key] = value.strftime('%Y-%m-%d')
                return result
            return obj
        
        class DateEncoder(json.JSONEncoder):
            """Custom JSON encoder that handles date/datetime objects"""
            def default(self, obj):
                if isinstance(obj, datetime) or isinstance(obj, date):
                    return obj.strftime('%Y-%m-%d')
                return super().default(obj)
        
        # Export genres
        with open(os.path.join(output_dir, "genres.json"), 'w', encoding='utf-8') as f:
            genres_dict = {str(k): obj_to_dict(v) for k, v in self.genres.items()}
            json.dump(genres_dict, f, indent=2, ensure_ascii=False, cls=DateEncoder)
        
        # Export directors
        with open(os.path.join(output_dir, "directors.json"), 'w', encoding='utf-8') as f:
            directors_dict = {str(k): obj_to_dict(v) for k, v in self.directors.items()}
            json.dump(directors_dict, f, indent=2, ensure_ascii=False, cls=DateEncoder)
        
        # Export characters
        with open(os.path.join(output_dir, "characters.json"), 'w', encoding='utf-8') as f:
            characters_dict = {str(k): obj_to_dict(v) for k, v in self.characters.items()}
            json.dump(characters_dict, f, indent=2, ensure_ascii=False, cls=DateEncoder)
        
        # Export actors
        with open(os.path.join(output_dir, "actors.json"), 'w', encoding='utf-8') as f:
            actors_dict = {str(k): obj_to_dict(v) for k, v in self.actors.items()}
            json.dump(actors_dict, f, indent=2, ensure_ascii=False, cls=DateEncoder)
        
        # Export movies
        with open(os.path.join(output_dir, "movies.json"), 'w', encoding='utf-8') as f:
            movies_dict = {str(k): obj_to_dict(v) for k, v in self.movies.items()}
            json.dump(movies_dict, f, indent=2, ensure_ascii=False, cls=DateEncoder)
        
        print(f"Exported all JSON data to {output_dir}/")
    
    def export_to_csv(self, output_dir="output"):
        """Export the generated data to CSV files"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Export genres
        with open(os.path.join(output_dir, "genres.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['id', 'name'])
            for genre_id, genre in self.genres.items():
                writer.writerow([
                    genre_id, 
                    self._safe_csv_string(genre.name)
                ])
        
        # Export directors
        with open(os.path.join(output_dir, "directors.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['id', 'name', 'birth_date', 'popularity', 'movie_ids'])
            for director_id, director in self.directors.items():
                writer.writerow([
                    director_id, 
                    self._safe_csv_string(director.name),
                    director.birth_date.strftime('%Y-%m-%d'),
                    director.popularity,
                    self._safe_csv_string(director.movie_ids)
                ])
        
        # Export characters
        with open(os.path.join(output_dir, "characters.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['id', 'name', 'description', 'movie_id'])
            for character_id, character in self.characters.items():
                writer.writerow([
                    character_id, 
                    self._safe_csv_string(character.name),
                    self._safe_csv_string(character.description),
                    character.movie_id
                ])
        
        # Export actors
        with open(os.path.join(output_dir, "actors.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['id', 'name', 'birth_date', 'popularity', 'character_ids', 'movie_ids'])
            for actor_id, actor in self.actors.items():
                writer.writerow([
                    actor_id, 
                    self._safe_csv_string(actor.name),
                    actor.birth_date.strftime('%Y-%m-%d'),
                    actor.popularity,
                    self._safe_csv_string(actor.character_ids),
                    self._safe_csv_string(actor.movie_ids)
                ])
        
        # Export movies
        with open(os.path.join(output_dir, "movies.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['id', 'title', 'release_year', 'duration', 'rating', 'director_id', 
                            'actor_ids', 'genre_ids', 'plot', 'budget', 'revenue'])
            for movie_id, movie in self.movies.items():
                writer.writerow([
                    movie_id, 
                    self._safe_csv_string(movie.title),
                    movie.release_year,
                    movie.duration,
                    movie.rating,
                    movie.director_id,
                    self._safe_csv_string(movie.actor_ids),
                    self._safe_csv_string(movie.genre_ids),
                    self._safe_csv_string(movie.plot),
                    movie.budget,
                    movie.revenue
                ])
        
        # Export movie-actor relationships (as a separate relationship table)
        with open(os.path.join(output_dir, "movie_actors.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['movie_id', 'actor_id'])
            for movie_id, movie in self.movies.items():
                for actor_id in movie.actor_ids:
                    writer.writerow([movie_id, actor_id])
        
        # Export movie-genre relationships (as a separate relationship table)
        with open(os.path.join(output_dir, "movie_genres.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['movie_id', 'genre_id'])
            for movie_id, movie in self.movies.items():
                for genre_id in movie.genre_ids:
                    writer.writerow([movie_id, genre_id])
        
        # Export actor-character relationships (as a separate relationship table)
        with open(os.path.join(output_dir, "actor_characters.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['actor_id', 'character_id'])
            for actor_id, actor in self.actors.items():
                for character_id in actor.character_ids:
                    writer.writerow([actor_id, character_id])
        
        print(f"Exported all CSV data to {output_dir}/")
    
    def export_all(self, output_dir="output"):
        """Export data to both JSON and CSV formats"""
        self.export_to_json(output_dir)
        self.export_to_csv(output_dir)
        print(f"All data exported to {output_dir}/")


# Example usage
if __name__ == "__main__":
    # Create a configuration with custom values
    config = GeneratorConfig(
        num_genres=18, # stick to the number of unique genres found in the movielens dataset, also don't use the "(no genres listed)" one 
        num_directors=501,
        num_actors=2001,
        num_characters=3250,
        num_movies=6500,
        min_actors_per_movie=3,
        max_actors_per_movie=11,
        min_genres_per_movie=1,
        max_genres_per_movie=7,
        use_real_data=True
    )
    
    # Create the generator
    generator = MovieDataGenerator(config)
    
    # Generate all data
    generator.generate_all_data()
    
    # Export to both JSON and CSV
    generator.export_all()
