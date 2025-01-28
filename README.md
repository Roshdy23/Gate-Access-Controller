# **Egyptian License Plate Recognization**

This project leverages OpenCV and Python to detect license plates from images. The program processes an image to identify the license plate area using various image processing techniques.

## Features:

- Image preprocessing to enhance the detection of license plates.
- Detection of license plates within an image.
- Developed a robust plate segmentation algorithm to accurately extract characters and digits from license plates under challenging conditions.
- Provides the option to select from various machine learning and deep learning models.
- Provides exclusive access to the designated vehicles only.


## Dataset:

The dataset used for this project is publicly available in the repository:
[Car Plates Dataset](https://github.com/Roshdy23/Gate-Access-Controller/tree/main/images)


## Installation:

Please ensure you have Python and the necessary libraries installed.
To install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
### Access Control:

To Run the Grant/Remove access app:
```bash
   python -m streamlit run .\access_control.py
```



### Run The Project

To run the project, use the following command:
```bash 
python -m streamlit run .\GUI.py
```
**Note**: make sure you have navigated the terminal to the location of the project folder.

### Pictures:
- access app:
  
![image](https://github.com/user-attachments/assets/850926de-d239-4515-896d-32b08db652fa)

- main app:

   - allowed vehicle example:
     
     ![image](https://github.com/user-attachments/assets/91450d1f-6fc9-4889-9569-0cf188891b15)
        ![image](https://github.com/user-attachments/assets/b79a25c9-10f5-4e56-be13-f792909c38e2)
    
   - not allowed vehicle example:
     
        ![image](https://github.com/user-attachments/assets/99af951a-35a1-45ac-b2a2-80fdff3290bc)
        ![image](https://github.com/user-attachments/assets/6a855730-839a-4ef9-883c-75d0e2b2763d)

   

## Future Enhancements

- **Real-Time Plate Recognition**: Enable real-time recognition by integrating camera feeds, eliminating the need to upload pictures manually and allowing instant access checks.
- **User Registration & Management**: Implement a comprehensive user registration system with profile management and authentication features..
- **Role-Based Access Control**: Introduce role-based access control to limit access based on user roles and permissions, enhancing security.
- **Mobile Application**: Develop a mobile version of the application for easy access and convenience, enabling users to manage and do an instant search with the mobile camera.
- **Cloud Integration**: Integrate the solution with cloud platforms for seamless scaling, and high availability.


##  Contributing

We welcome contributions to enhance the project. Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature.
3. Submit a pull request with detailed explanations.




