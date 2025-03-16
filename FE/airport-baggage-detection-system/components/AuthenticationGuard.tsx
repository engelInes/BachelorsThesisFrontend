import { useAuth } from "@/context/AuthenticationContext";
import { usePathname, useRouter } from "expo-router";
import React, { useEffect } from "react";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";

type AuthGuardProps = {
  children: React.ReactNode;
};

export default function AuthGuard({ children }: AuthGuardProps) {
  const { user, isLoading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (isLoading) return;

    const authRoutes = ["/", "/login", "/signup"];
    const isAuthRoute = authRoutes.includes(pathname);

    if (!user && !isAuthRoute) {
      router.replace("/");
    } else if (user && isAuthRoute) {
      router.replace("/(tabs)");
    }
  }, [user, isLoading, pathname]);

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#ffd33d" />
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }

  return <>{children}</>;
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#25292e",
  },
  loadingText: {
    marginTop: 10,
    color: "white",
    fontSize: 16,
  },
});
