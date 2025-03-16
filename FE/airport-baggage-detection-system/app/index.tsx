import { useAuth } from "@/context/AuthenticationContext";
import { Link, router } from "expo-router";
import { useEffect } from "react";
import { StyleSheet, Text, View } from "react-native";

export default function LandingScreen() {
  const { user, isLoading } = useAuth();

  useEffect(() => {
    if (!isLoading) {
      if (user) {
        router.replace("/(tabs)");
      } else {
        router.replace("/login");
      }
    }
  }, [user, isLoading]);

  return (
    <View style={styles.container}>
      <View style={styles.logoContainer}>
        <Text style={styles.logoText}>Sticker Smash</Text>
        <Text style={styles.tagline}>Create and share custom stickers</Text>
      </View>

      <View style={styles.buttonContainer}>
        <Link href="/(tabs)/login" style={[styles.button, styles.loginButton]}>
          <Text style={styles.loginButtonText}>Log In</Text>
        </Link>

        <Link
          href="/(tabs)/signup"
          style={[styles.button, styles.signupButton]}
        >
          <Text style={styles.signupButtonText}>Sign Up</Text>
        </Link>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#25292e",
    padding: 20,
  },
  logoContainer: {
    alignItems: "center",
    marginBottom: 80,
  },
  logoText: {
    fontSize: 36,
    fontWeight: "bold",
    color: "#ffd33d",
    marginBottom: 10,
  },
  tagline: {
    fontSize: 18,
    color: "white",
    textAlign: "center",
  },
  buttonContainer: {
    width: "100%",
  },
  button: {
    width: "100%",
    height: 56,
    borderRadius: 8,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 16,
  },
  loginButton: {
    backgroundColor: "#ffd33d",
  },
  signupButton: {
    backgroundColor: "transparent",
    borderWidth: 2,
    borderColor: "#ffd33d",
  },
  loginButtonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#25292e",
  },
  signupButtonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#ffd33d",
  },
});
