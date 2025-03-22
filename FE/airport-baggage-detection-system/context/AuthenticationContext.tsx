import AsyncStorage from "@react-native-async-storage/async-storage";
import React, { createContext, useContext, useEffect, useState } from "react";

const API_URL = "http://192.168.1.134:5000/api";

type User = {
  id: string;
  username: string;
  email: string;
  accountType: "user" | "admin";
};

type AuthContextType = {
  user: User | null;
  token: string | null;
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
  getUserImages: () => Promise<any[]>;
  uploadImage: (uri: string) => Promise<void>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadStoredData = async () => {
      try {
        const [storedToken, storedUser] = await Promise.all([
          AsyncStorage.getItem("token"),
          AsyncStorage.getItem("user"),
        ]);

        if (storedToken && storedUser) {
          setToken(storedToken);
          setUser(JSON.parse(storedUser));
        }
      } catch (error) {
        console.log("Error retrieving stored data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadStoredData();
  }, []);

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to login");
      }

      const { token: authToken, user: userData } = data;

      await Promise.all([
        AsyncStorage.setItem("token", authToken),
        AsyncStorage.setItem("user", JSON.stringify(userData)),
      ]);

      setToken(authToken);
      setUser(userData);
    } catch (error) {
      console.log("Login error:", error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const signup = async (
    username: string,
    email: string,
    password: string,
    accountType: "user" | "admin"
  ) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/signup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, email, password, accountType }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to signup");
      }

      const { token: authToken, user: userData } = data;

      await Promise.all([
        AsyncStorage.setItem("token", authToken),
        AsyncStorage.setItem("user", JSON.stringify(userData)),
      ]);

      setToken(authToken);
      setUser(userData);
    } catch (error) {
      console.log("Signup error:", error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    setIsLoading(true);
    try {
      await Promise.all([
        AsyncStorage.removeItem("token"),
        AsyncStorage.removeItem("user"),
      ]);

      setToken(null);
      setUser(null);
      console.log("logged out");
      return Promise.resolve();
    } catch (error) {
      console.log("Error during logout:", error);
      throw new Error("Failed to logout");
    } finally {
      setIsLoading(false);
    }
  };

  const isAdmin = () => {
    return user?.accountType === "admin";
  };

  const getUserImages = async () => {
    if (!token) {
      throw new Error("Not authenticated");
    }

    try {
      const response = await fetch(`${API_URL}/user/images`, {
        headers: {
          "x-auth-token": token,
        },
      });

      if (!response.ok) {
        throw new Error("Failed to fetch images");
      }

      return await response.json();
    } catch (error) {
      console.error("Error fetching user images:", error);
      throw error;
    }
  };

  const uploadImage = async (uri: string) => {
    if (!token || !user) {
      throw new Error("Not authenticated");
    }

    try {
      // Create form data for file upload
      const formData = new FormData();
      const filename = uri.split("/").pop() || "image.jpg";
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : "image/jpeg";

      // @ts-ignore
      formData.append("image", {
        uri,
        name: filename,
        type,
      });

      console.log("Uploading image:", uri);
      console.log("FormData:", JSON.stringify(formData));
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        headers: {
          "x-auth-token": token,
          //"Content-Type": "multipart/form-data",
        },
        body: formData,
      });

      console.log("upload response status:", response.status);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Upload failed");
      }

      return await response.json();
    } catch (error) {
      console.error("Error uploading image:", error);
      throw error;
    }
  };
  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isLoading,
        login,
        signup,
        logout,
        isAdmin,
        getUserImages,
        uploadImage,
      }}
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
