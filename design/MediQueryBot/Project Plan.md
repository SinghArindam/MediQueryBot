

# MediQuery chatbot
MediQuery consultant bot TTS LLM STT + fake utterance like sesame

[Reference 01](Reference01.md)

[MVP](MVP.md)

*   **Step 1: Start from Your Goal**
	* Making this project for **resume** project for **placement**
		* will have to design APIs so that i can use this for my **B.Tech project** as well in future
	* This project is for all ages and categories of people who want to consult a doctor but don't want to go to a hospital for small consultations
		* in future add sensors also so that accurate medical diagnosis can be achieved
	* simple interface, pleasant UI, and most importantly speech to speech teleconsultation, also fine tuning of model LLM, and comparison of different methods within finetuning, RAG etc.

*   **Step 2: Write User Stories**
	* user can register first which will generate a medical card with QR code
	* the user can use this card / QR code to scan and login using camera to app directly
	* user can login using username password as registered
	* option that shows multiple already logged in users for one click sign in or sign in with pin
	* use it as a guest - no consultation/chat saving
	* logged in usage saves chats and consultations (in form of chats)

	* after logging in there should be an option to directly simulate call with teleconsultation bot, in the center
	* chat should be on left side
	* a card on right side - function - TBD -blog or news?
	* a header of app on top
	* a section for user info display on top
	* Made by Arindam Singh at footer
	
	* a new page for chat used in v1, with option to select model finetuned LLM from dropdown and consult with it
	* a header of app on top
	* a section for user info display on top
	* Made by Arindam Singh at footer
	
	* possibly face authentication in the future

*   **Step 3: Define Your Data Models**
	* User
		* User ID (16digit number)
		* Full Name
			* First Name
			* Middle Name
			* Last Name
		* DOB
		* Image URL
		* Chats = {"Chat IDs",}
		* Unique Hash - Convert to QR
		* Height
		* Weight
		* BMI - Auto Calculate
		* City
		* Country
		* Email
		* Phone No
		* PIN for one click login
    *   Chats
	    * Chat ID
	    * Chat Name
	    * User ID - Optional
	    * Conversation = ```{
						Bot : {
						    type : "bot/user" - optional,
						    response : "",
						    response_time : "Responded in Xs Seconds.",
						    Reasoning : "Reasoning",
						    },
					    User : {
						    type : "bot/user" - optional,
						    msg : Hi",
						    time_stamp : "2025-06-29 16:04:03",
						    }, 
					    Bot : {
						    type : "bot/user" - optional,
						    response : "",
						    response_time : "Responded in Xs Seconds.",
						    Reasoning : "Reasoning",
						    },
					    }```
	* Medical Card
		* *User ID*
		* *Full Name*
			* *First Name*
			* *Middle Name*
			* *Last Name*
		* *DOB*
		* *Image URL*
		* *Unique Hash - Convert to QR*
		* *Height*
		* *Weight*
		* *BMI - Auto Calculate*
		* *City*
		* *Country*
		* *Email*
		* *Phone No*
		* Card Generation Time : 2025-06-29 16:04:03
    
    * User can access chats, stored and referenced as chat IDs
    * one user can have many chats
    * one chat only belongs to one user
    * Medical card is generated using user info and a copy of that is stored as medical data, it also has a generation timestamp

*   **Step 4: Nail an MVP (Minimum Viable Product)**
	* For MVP
	* **->V1<-** 
	* Features:
		* a new page for chat used in v1, with option to select model finetuned LLM from dropdown and consult/chat with it
		* a header of app on top
		* a section for user info display on top
		* Made by Arindam Singh at footer
	* Data Model:
		* Chats
		    * Chat ID
		    * Chat Name
		    * Conversation = ```{
							Bot : {
							    response : "",
							    response_time : "Responded in Xs Seconds.",
							    Reasoning : "Reasoning",
							    },
						    User : {
							    msg : Hi",
							    time_stamp : "2025-06-29 16:04:03",
							    }, 
						    Bot : {
							    response : "",
							    response_time : "Responded in Xs Seconds.",
							    Reasoning : "Reasoning",
							    },
						    }
    * 
	* **->V2<-**
	* Features:
		* after logging in there should be an option to directly simulate call with teleconsultation bot, in the center
		* chat should be on left side
		* a card on right side - function - TBD -blog or news?
		* a header of app on top
		* a section for user info display on top
		* Made by Arindam Singh at footer
	* Data Model:
		* Chats
		    * Chat ID
		    * Chat Name
		    * Conversation = ```{
							Bot : {
							    response : "",
							    response_time : "Responded in Xs Seconds.",
							    Reasoning : "Reasoning",
							    },
						    User : {
							    msg : Hi",
							    time_stamp : "2025-06-29 16:04:03",
							    }, 
						    Bot : {
							    response : "",
							    response_time : "Responded in Xs Seconds.",
							    Reasoning : "Reasoning",
							    },
						    }
	* **->V3<-**
	* Features:
		* user can register first which will generate a medical card with QR code
		* the user can use this card / QR code to scan and login using camera to app directly
		* user can login using username password as registered
		* option that shows multiple already logged in users for one click sign in or sign in with pin
		* use it as a guest - no consultation/chat saving
		* logged in usage saves chats and consultations (in form of chats)
		* a section for user info display on top
		* Made by Arindam Singh at footer
	* Data Model:
		* User
			* User ID (16digit number)
			* Full Name
				* First Name
				* Middle Name
				* Last Name
			* DOB
			* Image URL
			* Chats = {"Chat IDs",}
			* Unique Hash - Convert to QR
			* Height
			* Weight
			* BMI - Auto Calculate
			* City
			* Country
			* Email
			* Phone No
			* PIN for one click login
		* Medical Card
			* *User ID*
			* *Full Name*
				* *First Name*
				* *Middle Name*
				* *Last Name*
			* *DOB*
			* *Image URL*
			* *Unique Hash - Convert to QR*
			* *Height*
			* *Weight*
			* *BMI - Auto Calculate*
			* *City*
			* *Country*
			* *Email*
			* *Phone No*
			* Card Generation Time : 2025-06-29 16:04:03

*   **Step 5: Draw a Stupid Simple Prototype or Wireframe**
    *   ***-->V1<--***<img src="assets\MediQuery_V1.png">
    * ***-->V1.1<--***<img src="assets\MediQuery_V1.1.png">
    * ***-->V2<--***  <img src="assets\MediQuery_V2.png"> 
    * ***-->V2.1<--***<img src="assets\MediQuery_V2.1.png">
    * ***-->V2.2<--***<img src="assets\MediQuery_V2.2.png">
    * ***-->V2.3<--***<img src="assets\MediQuery_V2.3.png">
    * ***-->V2.4<--***<img src="assets\MediQuery_V2.4.png">

*   **Step 6: Understand the Future of the Project**
    *  Might need to scale to thousands of users. -> will handle in later versions, V5&6
    *  Something to tune for a long time.
    *  Real users will be on it.
    
    * for now lets use mongodb atlas for dbms, html+tailwind, etc using cdn for frontend, python + FastAPI for backend, huggingface, llama.cpp, and gemini API (google generative API), for LLM (incl. custom fine tuned on medical reasoning o1)

*   **Step 7: Drill into Specific Components**
	* for starters HTML standalone using CDN + python backend + MongoDB Atlas + Backend APIs
	* Later frontend port to react + rest same + DB as per scaling requirement
	* frontend + backend served using FastAPI on HuggingFace v2CPU

*   **Step 8: Pick Your Stack**
    *   Involves:
        *   A front-end technology - first HTML, then port to react later
        *   A back-end technology - Python + FastAPI
        *   A database - MongoDB later check based on scalability
        *   Specific frameworks - HuggingFace, Transformers, python-llama-cpp, google-generative-ai
    *   **Deployment path** : Hugging Face Spaces
    *   deployment won't take longer than coding the project.
    
    * Later deploy backend only on hf.co and host react app on vercel, or try on hf.co only

*   **Step 9: Overall Development Process**
    *   Steps :
        *   **Create a project skeleton:** Set up basic folder structure, development environments, and version control.
        *   **Set up the database and create data models:** Get the data layer working first, as data is crucial and other components depend on it.
        *   **Build backend routes:** Create API endpoints handling data operations and test them thoroughly before the front end. This gets core functionality working without worrying about UI.
        *   **Work on the front-end interface:** Connect it to the backend [9]. Since the backend works, focus can be on user experience and UI rather than debugging multiple stages simultaneously [9].
        *   **Project iteration:** Gradually add features, test as you go, and deploy early and often, even if not 100% complete [9].
        *   **Add automated deployments and testing:** If necessary for project goals [9].

License Arindam Singh (@SinghArindam)