import { FontAwesome } from "@expo/vector-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

type Props = {
  label: string;
  theme?: "primary";
  onPress?: () => void;
  disabled?: boolean;
};

export default function Button({
  label,
  theme,
  onPress,
  disabled = false,
}: Props) {
  if (theme === "primary") {
    return (
      <View style={[styles.buttonContainer, disabled && styles.buttonDisabled]}>
        <Pressable
          style={[styles.button, { backgroundColor: "#ffd33d" }]}
          onPress={onPress}
          disabled={disabled}
        >
          <FontAwesome
            name="picture-o"
            size={18}
            color="#25292e"
            style={styles.buttonIcon}
          />
          <Text style={[styles.buttonLabel, { color: "#25292e" }]}>
            {label}
          </Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={[styles.buttonContainer, disabled && styles.buttonDisabled]}>
      <Pressable style={styles.button} onPress={onPress} disabled={disabled}>
        <Text style={styles.buttonLabel}>{label}</Text>
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
  buttonDisabled: {
    opacity: 0.5,
  },
});
