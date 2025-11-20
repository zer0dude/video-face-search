import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import cv2

class FaceIdentifier:
    """
    Handles the logic for enrolling a person (creating a reference embedding)
    and identifying that person in new images using Facenet-PyTorch.

    Attributes:
        device (torch.device): The device (CPU or CUDA) used for inference.
        mtcnn (MTCNN): The face detection model.
        resnet (InceptionResnetV1): The face recognition model (embedding generator).
        known_embedding (np.ndarray): The averaged embedding of the enrolled person.
        is_enrolled (bool): Status flag indicating if a person has been enrolled.
    """
    def __init__(self):
        # Use CPU for stability in demo environment, or CUDA if available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # MTCNN for face detection
        self.mtcnn = MTCNN(
            keep_all=True, 
            device=self.device,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7]
        )
        
        # InceptionResnetV1 for face recognition (embeddings)
        # Load from local file to avoid download errors
        self.resnet = InceptionResnetV1(pretrained=None).eval().to(self.device)
        
        weights_path = 'assets/vggface2.pt'
        try:
            state_dict = torch.load(weights_path)
            # strict=False allows ignoring 'logits' keys which are present in the file but not in the model (since classify=False)
            self.resnet.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading local weights: {e}. Falling back to download (might fail).")
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.known_embedding = None
        self.is_enrolled = False

    def enroll_person(self, image_inputs):
        """
        Enrolls a person by generating embeddings from a list of reference images.
        
        The function detects the largest face in each image, generates an embedding,
        and then averages all valid embeddings to create a robust 'master' embedding.

        Args:
            image_inputs (list): A list of image paths (str), numpy arrays, or file-like objects.

        Returns:
            dict: A dictionary containing:
                - success (bool): Whether enrollment was successful.
                - message (str): Status message.
                - quality_score (float): A heuristic score (0.0-1.0) based on the number of valid images.
        """
        embeddings = []
        
        for img_input in image_inputs:
            try:
                # Convert to PIL Image
                image = None
                if isinstance(img_input, str):
                    image = Image.open(img_input).convert('RGB')
                elif isinstance(img_input, np.ndarray):
                    image = Image.fromarray(img_input)
                else:
                    # Assume file-like object (Streamlit UploadedFile)
                    # We might need to reset the pointer if it was read before, but usually fine.
                    image = Image.open(img_input).convert('RGB')

                # Detect faces and get crops
                # mtcnn(image) returns a list of tensors if keep_all=True
                # But we want to ensure we get the face.
                # Let's use .detect() to check first, or just run it.
                
                # Get cropped faces (tensors)
                faces_tensors = self.mtcnn(image)
                
                if faces_tensors is not None and len(faces_tensors) > 0:
                    # Take the first face found (assuming single person enrollment photos)
                    # faces_tensors is a tensor of shape (N, 3, 160, 160)
                    face_tensor = faces_tensors[0].unsqueeze(0).to(self.device)
                    
                    # Calculate embedding
                    emb = self.resnet(face_tensor).detach().cpu().numpy()
                    embeddings.append(emb)
            except Exception as e:
                print(f"Error enrolling image: {e}")
                continue

        if not embeddings:
            return {
                'success': False,
                'message': "No valid faces found in the provided images.",
                'quality_score': 0.0
            }

        # Create the master embedding by averaging
        # embeddings is a list of (1, 512) arrays
        self.known_embedding = np.mean(embeddings, axis=0)
        self.is_enrolled = True
        
        quality_score = min(len(embeddings) * 0.2, 1.0)
        
        return {
            'success': True,
            'message': f"Successfully enrolled with {len(embeddings)} valid images.",
            'quality_score': quality_score
        }

    def identify(self, frame, tolerance=0.8):
        """
        Identifies the enrolled person in a given video frame.

        Args:
            frame (np.ndarray): The video frame (RGB).
            tolerance (float): The cosine distance threshold for a match. 
                               Lower values are stricter. Default is 0.8.

        Returns:
            list: A list of bounding boxes (top, right, bottom, left) for matched faces.
                  Returns an empty list if no match is found.
        """
        if not self.is_enrolled:
            return []

        # Convert frame (numpy RGB) to PIL
        image = Image.fromarray(frame)

        # Detect faces
        boxes, _ = self.mtcnn.detect(image)
        
        if boxes is None:
            return []
            
        # Get embeddings
        # We need the face tensors. 
        faces_tensors = self.mtcnn(image)
        
        if faces_tensors is None:
            return []

        matches_locations = []
        
        # Calculate embeddings for all faces found
        embeddings = self.resnet(faces_tensors.to(self.device)).detach().cpu().numpy()
        
        for i, box in enumerate(boxes):
            # Calculate Euclidean distance
            dist = np.linalg.norm(embeddings[i] - self.known_embedding)
            
            # Check match
            if dist < tolerance:
                # Box is [x1, y1, x2, y2]
                x1, y1, x2, y2 = box
                
                # Convert to (top, right, bottom, left) -> (y1, x2, y2, x1)
                matches_locations.append((int(y1), int(x2), int(y2), int(x1)))

        return matches_locations
