import {
  addDoc,
  collection,
  deleteDoc,
  doc,
  getDocs,
  query,
  serverTimestamp,
  where,
} from "firebase/firestore";
import {
  deleteObject,
  getDownloadURL,
  getStorage,
  ref,
  uploadBytes,
} from "firebase/storage";
import { db } from "../context/AuthenticationContext";

const storage = getStorage();

export type StoredImage = {
  id: string;
  userId: string;
  username: string;
  imageUrl: string;
  createdAt: Date;
  hasEmoji: boolean;
};

export const uploadImage = async (
  uri: string,
  userId: string,
  username: string,
  hasEmoji: boolean = false
): Promise<string> => {
  try {
    const response = await fetch(uri);
    const blob = await response.blob();

    const filename = `stickers/${userId}/${Date.now()}.jpg`;
    const storageRef = ref(storage, filename);

    const snapshot = await uploadBytes(storageRef, blob);

    const downloadURL = await getDownloadURL(snapshot.ref);

    await addDoc(collection(db, "images"), {
      userId,
      username,
      imageUrl: downloadURL,
      createdAt: serverTimestamp(),
      hasEmoji,
    });

    return downloadURL;
  } catch (error) {
    console.error("Error uploading image:", error);
    throw error;
  }
};

export const getUserImages = async (userId: string): Promise<StoredImage[]> => {
  try {
    const q = query(collection(db, "images"), where("userId", "==", userId));

    const querySnapshot = await getDocs(q);

    const images: StoredImage[] = [];

    querySnapshot.forEach((doc) => {
      const data = doc.data();
      images.push({
        id: doc.id,
        userId: data.userId,
        username: data.username,
        imageUrl: data.imageUrl,
        createdAt: data.createdAt?.toDate() || new Date(),
        hasEmoji: data.hasEmoji || false,
      });
    });

    return images.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  } catch (error) {
    console.error("Error fetching user images:", error);
    throw error;
  }
};

export const deleteImage = async (
  imageId: string,
  imageUrl: string
): Promise<void> => {
  try {
    await deleteDoc(doc(db, "images", imageId));

    const storageRef = ref(storage, imageUrl);
    await deleteObject(storageRef);
  } catch (error) {
    console.error("Error deleting image:", error);
    throw error;
  }
};
