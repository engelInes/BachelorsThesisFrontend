import Button from "@/components/Buttons";
import CircleButton from "@/components/CircleButton";
import EmojiList from "@/components/EmojiList";
import EmojiPicker from "@/components/EmojiPicker";
import EmojiSticker from "@/components/EmojiSticker";
import IconButton from "@/components/IconButton";
import ImageViewer from "@/components/ImageViewer";
import { useAuth } from "@/context/AuthenticationContext";
import domtoimage from "dom-to-image";
import { type ImageSource } from "expo-image";
import * as ImagePicker from "expo-image-picker";
import * as MediaLibrary from "expo-media-library";
import { router } from "expo-router";
import { useEffect, useRef, useState } from "react";
import { Alert, Platform, StyleSheet, View } from "react-native";
import { captureRef } from "react-native-view-shot";

const PlaceholderImage = require("../../assets/images/image.png");

export default function Index() {
  const { user, isLoading } = useAuth();
  const imageRef = useRef<View>(null);
  const [status, requestPermisiion] = MediaLibrary.usePermissions();
  const [selectedImage, setSelectedImage] = useState<string | undefined>(
    undefined
  );
  const { uploadImage } = useAuth();
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [showAppOptions, setShowAppOption] = useState<boolean>(false);
  const [isModalVisible, setIsModalVisible] = useState<boolean>(false);
  const [pickedEmoji, setPickedEmoji] = useState<ImageSource | undefined>(
    undefined
  );
  useEffect(() => {
    if (!isLoading && !user) {
      router.replace("/");
    }
  }, [user, isLoading]);

  if (isLoading || !user) {
    return null;
  }

  const pcikImageAsync = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      console.log(result);
    } else {
      alert("Please select an image");
    }
  };

  const takePhotoAsync = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();

    if (status !== "granted") {
      alert("we need camera permissions");
      return;
    }

    let result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      setShowAppOption(true);
    } else {
      alert("You did not take any photo");
    }
  };

  const onReset = () => {
    setShowAppOption(false);
  };

  const onAddSticker = () => {
    setIsModalVisible(true);
  };

  const onModalClose = () => {
    setIsModalVisible(false);
  };

  const onSaveImageAsync = async () => {
    setIsSaving(true);
    if (Platform.OS !== "web") {
      try {
        const localUri = await captureRef(imageRef, {
          height: 440,
          quality: 1,
        });

        await MediaLibrary.saveToLibraryAsync(localUri);
        if (localUri) {
          try {
            await uploadImage(localUri);
            Alert.alert(
              "Success",
              "Image saved locally and uploaded to your account!"
            );
          } catch (error) {
            console.error("Upload error:", error);
            Alert.alert(
              "Partially Saved",
              "Image saved to your device but failed to upload to your account."
            );
          }
        }
      } catch (e) {
        console.log(e);
      } finally {
        setIsSaving(false);
      }
    } else {
      try {
        // @ts-ignore
        const dataUrl = await domtoimage.toJpeg(imageRef.current, {
          quality: 0.95,
          width: 320,
          height: 440,
        });

        let link = document.createElement("a");
        link.download = "sticker-smash.jpeg";
        link.href = dataUrl;
        link.click();

        const response = await fetch(dataUrl);
        const blob = await response.blob();
        const file = new File([blob], "sticker-smash.jpeg", {
          type: "image/jpeg",
        });

        const formData = new FormData();
        formData.append("image", file);

        Alert.alert("Success", "Image saved!");
      } catch (e) {
        console.log(e);
      }
    }
  };

  if (status == null) {
    requestPermisiion();
  }
  return (
    <View style={styles.container}>
      <View style={styles.imageContainer}>
        <View ref={imageRef} collapsable={false}>
          <ImageViewer imgSource={selectedImage || PlaceholderImage} />
          {pickedEmoji && (
            <EmojiSticker imageSize={40} stickerSource={pickedEmoji} />
          )}
        </View>
      </View>
      {showAppOptions ? (
        <View style={styles.optionsContainer}>
          <View style={styles.optionsRow}>
            <IconButton icon="refresh" label="Reset" onPress={onReset} />
            <CircleButton onPress={onAddSticker} />
            <IconButton
              icon="save-alt"
              label="Save"
              onPress={onSaveImageAsync}
            />
          </View>
        </View>
      ) : (
        <View style={styles.footerContainer}>
          <Button
            onPress={pcikImageAsync}
            label="Choose photo"
            theme="primary"
          />
          <Button onPress={takePhotoAsync} label="Take photo" theme="primary" />
          <Button
            label="Use this photo"
            onPress={() => setShowAppOption(true)}
          />
        </View>
      )}
      ;
      <EmojiPicker isVisible={isModalVisible} onClose={onModalClose}>
        <EmojiList onSelect={setPickedEmoji} onCloseModal={onModalClose} />
      </EmojiPicker>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#25292e",
  },
  text: {
    color: "white",
  },
  button: {
    fontSize: 20,
    textDecorationLine: "underline",
    color: "#fff",
  },
  image: {
    width: 320,
    height: 440,
    borderRadius: 18,
  },
  imageContainer: {
    flex: 1,
  },
  footerContainer: {
    flex: 1 / 3,
    alignItems: "center",
    justifyContent: "space-evenly",
  },
  optionsContainer: {
    position: "absolute",
    bottom: 80,
  },
  optionsRow: {
    alignItems: "center",
    flexDirection: "row",
  },
});
