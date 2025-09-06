Persona: You are a senior React developer specializing in creating user-friendly, responsive, and well-documented applications. Your task is to implement a professional-grade frontend for a healthcare EDI summarization API.

Task: Create a complete, self-contained React web application named ragX12-app. This application will serve as a user-friendly interface for the provided API, enabling non-technical users to easily summarize X12 files.

Requirements:

UI/UX:

Design a clean, modern, and accessible interface suitable for healthcare or billing professionals.

Incorporate a "glass design" aesthetic in a light mode theme. This involves using semi-transparent, blurred backgrounds for cards and containers and subtle borders.

The layout must be fully responsive, adapting gracefully to mobile, tablet, and desktop screen sizes.

Use functional components and modern React hooks (useState, useEffect).

Style the application using Tailwind CSS. The Tailwind CDN script must be included in the HTML to ensure the styles work out-of-the-box.

Prioritize readability by ensuring all text has a strong contrast against the blurred background.

Core Functionality:

Health Check: On component mount, automatically ping the /health API endpoint. Display the status prominently (e.g., "API Status: OK" or "API Status: Offline").

Input Form:

Provide a large, resizable textarea for users to paste or type raw X12 EDI content.

Include a submission button to trigger the summarization process.

API Communication:

When the form is submitted, make a POST request to the /summarize endpoint. The request body should match the specified API reference: { "x12": "<raw X12 string>", "top_k": 6, "include_prompt": false }.

Set include_prompt to true to demonstrate the capability, and top_k can be a hardcoded value of 6.

Output Display:

Clearly and prominently display the human-readable summary returned from the API.

Provide optional, collapsible sections to show the parsed_segments, retrieval_context, and prompt for transparency.

State Management:

Use React hooks to manage the application's state, including the input text, the summary and other API response data, loading state, and any error messages.

Error & Loading Handling:

Display a clear "Loading..." or spinner animation while an API request is in progress.

Gracefully handle and display error messages in a user-friendly format if an API call fails.

Code Quality:

Include comprehensive code comments to explain the purpose of each section, state variable, and function.

Use a placeholder API URL (e.g., https://api.example.com) to make the code runnable without a live backend.