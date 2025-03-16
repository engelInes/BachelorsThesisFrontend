import { auth, db } from "@/FirebaseConfig";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut,
} from "firebase/auth";
import { doc, getDoc, setDoc } from "firebase/firestore";
import React, { createContext, useContext, useEffect, useState } from "react";
type User = {
  uid: string;
  username: string;
  email: string;
  accountType: "user" | "admin";
};

type AuthContextType = {
  user: User | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (
    username: string,
    email: string,
    password: string,
    accountType: "user" | "admin"
  ) => Promise<void>;
  logout: () => Promise<void>;
  isAdmin: () => boolean;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  //   useEffect(() => {
  //     const checkUser = async () => {
  //       try {
  //         const userData = await AsyncStorage.getItem("user");
  //         if (userData) {
  //           setUser(JSON.parse(userData));
  //         }
  //       } catch (error) {
  //         console.log("Error retrieving user data:", error);
  //       } finally {
  //         setIsLoading(false);
  //       }
  //     };

  //     checkUser();
  //   }, []);
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        try {
          const userDoc = await getDoc(doc(db, "users", firebaseUser.uid));

          if (userDoc.exists()) {
            const userData = userDoc.data() as Omit<User, "uid">;
            const user: User = {
              uid: firebaseUser.uid,
              username: userData.username,
              email: userData.email,
              accountType: userData.accountType,
            };

            await AsyncStorage.setItem("user", JSON.stringify(user));
            setUser(user);
          } else {
            await auth.signOut();
            await AsyncStorage.removeItem("user");
            setUser(null);
          }
        } catch (error) {
          console.error("Error fetching user data:", error);
          try {
            const cachedUser = await AsyncStorage.getItem("user");
            if (cachedUser) {
              setUser(JSON.parse(cachedUser));
            }
          } catch (asyncError) {
            console.error("Error getting cached user:", asyncError);
          }
        }
      } else {
        await AsyncStorage.removeItem("user");
        setUser(null);
      }
      setIsLoading(false);
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    const checkUser = async () => {
      try {
        const userData = await AsyncStorage.getItem("user");
        if (userData && !user) {
          setUser(JSON.parse(userData));
        }
      } catch (error) {
        console.log("Error retrieving cached user data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    checkUser();
  }, []);

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    try {
      // Sign in with Firebase
      const userCredential = await signInWithEmailAndPassword(
        auth,
        email,
        password
      );
      const firebaseUser = userCredential.user;

      // Get user data from Firestore
      const userDoc = await getDoc(doc(db, "users", firebaseUser.uid));

      if (userDoc.exists()) {
        const userData = userDoc.data() as Omit<User, "uid">;
        const user: User = {
          uid: firebaseUser.uid,
          username: userData.username,
          email: userData.email,
          accountType: userData.accountType,
        };

        await AsyncStorage.setItem("user", JSON.stringify(user));
        setUser(user);
      } else {
        throw new Error("User data not found");
      }
    } catch (error: any) {
      console.error("Login error:", error);
      throw new Error(error.message || "Failed to login");
    } finally {
      setIsLoading(false);
    }
    // const mockUser: User = {
    //   id: "123",
    //   username: email.split("@")[0],
    //   email,
    //   accountType: "user",
    // };

    // try {
    //   await AsyncStorage.setItem("user", JSON.stringify(mockUser));
    //   setUser(mockUser);
    // } catch (error) {
    //   console.log("Error saving user data:", error);
    //   throw new Error("Failed to login");
    // } finally {
    //   setIsLoading(false);
    // }
  };

  const signup = async (
    username: string,
    email: string,
    password: string,
    accountType: "user" | "admin"
  ) => {
    setIsLoading(true);
    try {
      // Create user in Firebase Auth
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email,
        password
      );
      const firebaseUser = userCredential.user;

      // Create user document in Firestore
      const userData: Omit<User, "uid"> = {
        username,
        email,
        accountType,
      };

      await setDoc(doc(db, "users", firebaseUser.uid), {
        ...userData,
        createdAt: new Date().toISOString(),
      });

      // Create user object with uid
      const newUser: User = {
        uid: firebaseUser.uid,
        ...userData,
      };

      // Store user data in AsyncStorage for offline access
      await AsyncStorage.setItem("user", JSON.stringify(newUser));
      setUser(newUser);
    } catch (error: any) {
      console.error("Signup error:", error);
      throw new Error(error.message || "Failed to signup");
    } finally {
      setIsLoading(false);
    }
    // const mockUser: User = {
    //   id: "123",
    //   username,
    //   email,
    //   accountType: "user",
    // };

    // try {
    //   await AsyncStorage.setItem("user", JSON.stringify(mockUser));
    //   setUser(mockUser);
    // } catch (error) {
    //   console.log("Error saving user data:", error);
    //   throw new Error("Failed to signup");
    // } finally {
    //   setIsLoading(false);
    // }
  };

  const logout = async () => {
    setIsLoading(true);
    try {
      // Sign out from Firebase
      await signOut(auth);
      // Remove user data from AsyncStorage
      await AsyncStorage.removeItem("user");
      setUser(null);
      console.log("logged out");
      return Promise.resolve();
    } catch (error) {
      console.error("Logout error:", error);
      throw new Error("Failed to logout");
    } finally {
      setIsLoading(false);
    }
    // try {
    //   await AsyncStorage.removeItem("user");
    //   setUser(null);
    //   console.log("logged out");
    //   return Promise.resolve();
    // } catch (error) {
    //   console.log("Error removing user data:", error);
    //   throw new Error("Failed to logout");
    // } finally {
    //   setIsLoading(false);
    // }
  };

  const isAdmin = () => {
    return user?.accountType === "admin";
  };

  return (
    <AuthContext.Provider
      value={{ user, isLoading, login, signup, logout, isAdmin }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

export { db };
