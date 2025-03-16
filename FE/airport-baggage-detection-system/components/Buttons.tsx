import { FontAwesome } from "@expo/vector-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

type Props = {
  label: string;
  theme?: "primary";
  onPress?: () => void;
};

export default function Button({ label, theme, onPress }: Props) {
  return (
    <View style={styles.buttonContainer}>
      <Pressable
        style={[styles.button, theme === "primary" && styles.primaryButton]}
        onPress={onPress}
      >
        {theme === "primary" && (
          <FontAwesome
            name="picture-o"
            size={18}
            color="#25292e"
            style={styles.buttonIcon}
          />
        )}
        <Text
          style={[
            styles.buttonLabel,
            theme === "primary" && styles.primaryButtonLabel,
          ]}
        >
          {label}
        </Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  buttonContainer: {
    width: 320,
    height: 68,
    marginHorizontal: 20,
    alignItems: "center",
    justifyContent: "center",
    padding: 3,
  },
  button: {
    borderRadius: 10,
    width: "100%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
  },
  buttonIcon: {
    paddingRight: 8,
  },
  buttonLabel: {
    color: "#fff",
  },
  primaryButtonLabel: {
    color: "#25292e",
  },
  primaryButton: {
    backgroundColor: "#fff",
  },
});
