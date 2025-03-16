// Import the functions you need from the SDKs you need
import ReactNativeAsyncStorage from "@react-native-async-storage/async-storage";
import { getAnalytics } from "firebase/analytics";
import { initializeApp } from "firebase/app";
import { initializeAuth } from "firebase/auth";
import getReactNativePersistence from "@react-native-firebase/auth";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCNpqjPS_8arIJU8rKi5rOjka1OQ8ZZASc",
  authDomain: "bt-airport-system-app.firebaseapp.com",
  projectId: "bt-airport-system-app",
  storageBucket: "bt-airport-system-app.firebasestorage.app",
  messagingSenderId: "399267183630",
  appId: "1:399267183630:web:76bad8df1c5765b52b505f",
  measurementId: "G-3XCNJT2CJ3",
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
export const auth = initializeAuth(app, {
  persistence: getReactNativePersistence(ReactNativeAsyncStorage),
});
const analytics = getAnalytics(app);
export const db = getFirestore(app);
