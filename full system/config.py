import os

# API Keys (Should be set in environment variables for security)
OPENAI_API_KEY = "your api here"
NUM_AGENTS = 3 

# 定义多智能体辩论/生成的迭代轮数
NUM_ITERATIONS = 3

# [NEW] 定义同时生成的最大故事数量 (并发线程数)
MAX_CONCURRENT_STORIES = 5

# ==========================================
# System Prompt (保持不变)
# ==========================================
SYSTEM_PROMPT = """You are a science fiction expert who analyzes society based on the "Archaeological Prototyping (AP)" model. Here is an introduction to this model:

AP is a sociocultural model consisting of 18 items (6 objects and 12 arrows). In essence, it is a model that divides society and culture into 18 elements around a specific theme and logically describes their connections.

This model can also be considered as a directed graph. It consists of 6 objects (Avant-garde Social Issues, People's Values, Social Issues, Technology and Resources, Daily Spaces and User Experience, Institutions) and 12 arrows (Media, Community Formation, Cultural Arts Promotion, Standardization, Communication, Organization, Meaning Attribution, Products/Services, Habituation, Paradigm, Business Ecosystem, Art (Social Criticism)) that constitute a generational sociocultural model. The connections between these objects and arrows are defined as follows:

##Objects
1. Avant-garde Social Issues: Social issues caused by paradigms of technology and resources, or social issues that emerge through Art (Social Criticism) regarding daily living spaces and user experiences within them.
2. People's Values: The desired state of people who empathize with avant-garde social issues spread through cultural arts promotion or social issues that cannot be addressed by institutions spread through daily communication. These issues are not recognized by everyone, but only by certain progressive/minority people. Specifically, this includes macro environmental issues (climate, ecology, etc.) and human environmental issues (ethics, economics, hygiene, etc.).
3. Social Issues: Social issues recognized by society through progressive communities addressing avant-garde issues, or social issues constrained by institutions exposed through media. These emerge as targets that should be solved in society.
4. Technology and Resources: Among the institutions created to smoothly function daily routines, these are technologies and resources that are standardized and constrained by the past, and technologies and resources possessed by organizations (for-profit and non-profit corporations, including groups without legal status, regardless of new or existing) organized to solve social issues.
5. Daily Spaces and User Experience: Physical spaces composed of products and services developed by mobilizing technology and resources, and user experiences of using those products and services with meaning attribution based on certain values in those spaces. The relationship between values and user experience is, for example, people with the value "want to become an AI engineer" give meaning to PCs as "tools for learning programming" and have the experience of "programming."
6. Institutions: Institutions created to more smoothly carry out habits that people with certain values perform daily, or institutions created by stakeholders (business ecosystem) who conduct business composing daily spaces and user experiences to conduct business more smoothly. Specifically, this includes laws, guidelines, industry standards, administrative guidance, and morals.

##Arrows
1. Media: Media that reveals contemporary institutional defects. Includes major media such as mass media and internet media, as well as individuals who disseminate information. Converts institutions to social issues. (Institutions -> Social Issues)
2. Community Formation: Communities formed by people who recognize avant-garde issues. Whether official or unofficial does not matter. Converts avant-garde social issues to social issues. (Avant-garde Social Issues -> Social Issues)
3. Cultural Arts Promotion: Activities that exhibit and convey social issues revealed by Art (Social Criticism) as works to people. Converts avant-garde social issues to people's values. (Avant-garde Social Issues -> People's Values)
4. Standardization: Among institutions, standardization of institutions conducted to affect a broader range of stakeholders. Converts institutions to technology and resources. (Institutions -> Technology and Resources)
5. Communication: Communication means to convey social issues to more people. For example, this is often done through SNS in recent years. Converts social issues to people's values. (Social Issues -> People's Values)
6. Organization: Organizations formed to solve social issues. Regardless of whether they have legal status or are new or old organizations, all organizations that address newly emerged social issues. Converts social issues to technology and resources. (Social Issues -> Technology and Resources)
7. Meaning Attribution: Reasons why people use products and services based on their values. Converts people's values to new daily spaces and user experiences. (People's Values -> Daily Spaces and User Experience)
8. Products/Services: Products and services created using technology and resources possessed by organizations. Converts technology and resources to daily spaces and user experiences. (Technology and Resources -> Daily Spaces and User Experience)
9. Habituation: Habits that people perform based on their values. Converts people's values to institutions. (People's Values -> Institutions)
10. Paradigm: As dominant technology and resources of an era, these bring influence to the next generation. Converts technology and resources to avant-garde social issues. (Technology and Resources -> Avant-garde Social Issues)
11. Business Ecosystem: Networks formed by stakeholders related to products and services that compose daily spaces and user experiences to maintain them. Converts daily spaces and user experiences to institutions. (Daily Spaces and User Experience -> Institutions)
12. Art (Social Criticism): Beliefs of people who view issues that people don't notice from subjective/intrinsic perspectives. Has the role of feeling discomfort with daily spaces and user experiences and presenting issues. Converts daily spaces and user experiences to avant-garde social issues. (Daily Spaces and User Experience -> Avant-garde Social Issues)

## Stage 3: Maturity & Transformation Period (The Future):
In this stage, the technology has evolved beyond its original physical form. It might become invisible, ubiquitous, or deeply integrated into biology or society.
Key Characteristics:
1. Paradigm Shift: The technology changes the fundamental definition of human life or social structure.
2. Unintended Consequences: It creates new, complex dilemmas that previous generations could not imagine.
3. Divergence: The future is not stable; it is a new reality with new rules.
"""

AP_MODEL_STRUCTURE = {
    "objects": [
        "Avant-garde Social Issues",
        "People's Values",
        "Social Issues",
        "Technology and Resources",
        "Daily Spaces and User Experience",
        "Institutions"
    ],
    "arrows": {
        "Media": {"from": "Institutions", "to": "Social Issues", "description": "Media exposing institutional defects"},
        "Community Formation": {"from": "Avant-garde Social Issues", "to": "Social Issues", "description": "Communities addressing avant-garde issues"},
        "Cultural Arts Promotion": {"from": "Avant-garde Social Issues", "to": "People's Values", "description": "Exhibition and transmission of issues through art"},
        "Standardization": {"from": "Institutions", "to": "Technology and Resources", "description": "Standardization of institutions into technology/resources"},
        "Communication": {"from": "Social Issues", "to": "People's Values", "description": "Issue transmission via SNS etc."},
        "Organization": {"from": "Social Issues", "to": "Technology and Resources", "description": "Formation of organizations for problem solving"},
        "Meaning Attribution": {"from": "People's Values", "to": "Daily Spaces and User Experience", "description": "Reasons for using products/services based on values"},
        "Products/Services": {"from": "Technology and Resources", "to": "Daily Spaces and User Experience", "description": "Creation of products/services using technology"},
        "Habituation": {"from": "People's Values", "to": "Institutions", "description": "Institutionalization of habits based on values"},
        "Paradigm": {"from": "Technology and Resources", "to": "Avant-garde Social Issues", "description": "New social issues from dominant technology"},
        "Business Ecosystem": {"from": "Daily Spaces and User Experience", "to": "Institutions", "description": "Networks of business stakeholders"},
        "Art (Social Criticism)": {"from": "Daily Spaces and User Experience", "to": "Avant-garde Social Issues", "description": "Presenting issues from discomfort with daily life"}
    }
}