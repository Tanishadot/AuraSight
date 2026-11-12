import 'package:flutter/material.dart';

/// AuraSight's palette: sage, forest green and warm ivory.
/// Every text/background pair used below meets WCAG AA (most meet AAA);
/// see test/theme_contrast_test.dart.
class AuraColors {
  static const forest = Color(0xFF1F3A2B); // primary on light
  static const forestDeep = Color(0xFF142619);
  static const sage = Color(0xFF8DAF94);
  static const sageSoft = Color(0xFFDCE8DD);
  static const sageInk = Color(0xFF3B5E45); // sage dark enough for text on ivory
  static const ivory = Color(0xFFF8F4EA);
  static const ivoryRaised = Color(0xFFFFFDF7);
  static const ink = Color(0xFF17241C);
  static const muted = Color(0xFF4E5C53);
  static const line = Color(0xFFD9D3C4);
  static const danger = Color(0xFF9E2A1E);
  static const dangerSoft = Color(0xFFF6DED9);

  // dark theme
  static const night = Color(0xFF0F1712);
  static const nightRaised = Color(0xFF18241C);
  static const nightLine = Color(0xFF2E3D33);
  static const sageLight = Color(0xFFA9C9AF);
  static const ivoryText = Color(0xFFF2EDE1);
  static const mutedDark = Color(0xFFB5C0B8);
  static const dangerDark = Color(0xFFFFB4A8);
}

/// Minimum touch target for primary controls (WCAG recommends >= 44 pt;
/// we use 56 dp for everyday controls and 72 dp for the main actions).
const double kTouchTarget = 56;
const double kHeroTarget = 72;

ThemeData auraTheme(Brightness brightness) {
  final dark = brightness == Brightness.dark;
  final scheme = dark
      ? const ColorScheme(
          brightness: Brightness.dark,
          primary: AuraColors.sageLight,
          onPrimary: AuraColors.forestDeep,
          primaryContainer: AuraColors.forest,
          onPrimaryContainer: AuraColors.ivoryText,
          secondary: AuraColors.sage,
          onSecondary: AuraColors.forestDeep,
          surface: AuraColors.night,
          onSurface: AuraColors.ivoryText,
          surfaceContainerHighest: AuraColors.nightRaised,
          onSurfaceVariant: AuraColors.mutedDark,
          outline: AuraColors.nightLine,
          error: AuraColors.dangerDark,
          onError: AuraColors.forestDeep,
        )
      : const ColorScheme(
          brightness: Brightness.light,
          primary: AuraColors.forest,
          onPrimary: AuraColors.ivory,
          primaryContainer: AuraColors.sageSoft,
          onPrimaryContainer: AuraColors.forestDeep,
          secondary: AuraColors.sageInk,
          onSecondary: AuraColors.ivory,
          surface: AuraColors.ivory,
          onSurface: AuraColors.ink,
          surfaceContainerHighest: AuraColors.ivoryRaised,
          onSurfaceVariant: AuraColors.muted,
          outline: AuraColors.line,
          error: AuraColors.danger,
          onError: AuraColors.ivory,
        );

  final base = ThemeData(colorScheme: scheme, useMaterial3: true, brightness: brightness);
  // Generous sizes; everything scales with the system text size (never clamped).
  final text = base.textTheme.copyWith(
    headlineMedium: const TextStyle(fontSize: 30, fontWeight: FontWeight.w700, height: 1.2),
    titleLarge: const TextStyle(fontSize: 22, fontWeight: FontWeight.w700, height: 1.25),
    titleMedium: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700, height: 1.3),
    bodyLarge: const TextStyle(fontSize: 20, height: 1.45),
    bodyMedium: const TextStyle(fontSize: 17, height: 1.45),
    labelLarge: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700, letterSpacing: 0.2),
  ).apply(bodyColor: scheme.onSurface, displayColor: scheme.onSurface);

  final shape = RoundedRectangleBorder(borderRadius: BorderRadius.circular(18));
  return base.copyWith(
    textTheme: text,
    scaffoldBackgroundColor: scheme.surface,
    materialTapTargetSize: MaterialTapTargetSize.padded,
    visualDensity: VisualDensity.standard,
    appBarTheme: AppBarTheme(
      backgroundColor: scheme.surface,
      foregroundColor: scheme.onSurface,
      elevation: 0,
      centerTitle: false,
      titleTextStyle: text.titleLarge,
    ),
    cardTheme: CardThemeData(
      color: scheme.surfaceContainerHighest,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(22),
        side: BorderSide(color: scheme.outline),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size(kTouchTarget, kTouchTarget),
        padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 14),
        textStyle: text.labelLarge,
        shape: shape,
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size(kTouchTarget, kTouchTarget),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        textStyle: text.labelLarge,
        foregroundColor: scheme.onSurface,
        side: BorderSide(color: scheme.onSurfaceVariant, width: 1.5),
        shape: shape,
      ),
    ),
    navigationBarTheme: NavigationBarThemeData(
      height: 76,
      backgroundColor: dark ? AuraColors.nightRaised : AuraColors.ivoryRaised,
      indicatorColor: dark ? AuraColors.forest : AuraColors.sageSoft,
      labelTextStyle: WidgetStatePropertyAll(text.labelLarge!.copyWith(fontSize: 14)),
      iconTheme: WidgetStatePropertyAll(IconThemeData(size: 28, color: scheme.onSurface)),
    ),
    switchTheme: SwitchThemeData(
      thumbColor: WidgetStateProperty.resolveWith(
          (s) => s.contains(WidgetState.selected) ? scheme.onPrimary : scheme.onSurfaceVariant),
      trackColor: WidgetStateProperty.resolveWith(
          (s) => s.contains(WidgetState.selected) ? scheme.primary : scheme.outline),
    ),
    sliderTheme: SliderThemeData(
      trackHeight: 6,
      activeTrackColor: scheme.primary,
      thumbColor: scheme.primary,
      overlayShape: const RoundSliderOverlayShape(overlayRadius: 28),
    ),
    listTileTheme: ListTileThemeData(
      minVerticalPadding: 14,
      titleTextStyle: text.titleMedium,
      subtitleTextStyle: text.bodyMedium!.copyWith(color: scheme.onSurfaceVariant),
    ),
    snackBarTheme: SnackBarThemeData(
      behavior: SnackBarBehavior.floating,
      contentTextStyle: text.bodyMedium!.copyWith(color: dark ? AuraColors.forestDeep : AuraColors.ivory),
      backgroundColor: dark ? AuraColors.sageLight : AuraColors.forest,
    ),
  );
}
