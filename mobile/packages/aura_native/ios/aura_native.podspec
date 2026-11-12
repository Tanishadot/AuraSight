#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint aura_native.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'aura_native'
  s.version          = '0.0.1'
  s.summary          = 'AuraSight native modules: speech, device state, on-device inference.'
  s.description      = <<-DESC
AuraSight native modules for iOS: AVSpeechSynthesizer speech and device thermal/memory state.
                       DESC
  s.homepage         = 'https://github.com/Tanishadot/AuraSight'
  s.license          = { :file => '../LICENSE' }
  s.author           = 'Tanisha Jogi'
  s.source           = { :path => '.' }
  s.source_files = 'aura_native/Sources/aura_native/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '15.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  # If your plugin requires a privacy manifest, for example if it uses any
  # required reason APIs, update the PrivacyInfo.xcprivacy file to describe your
  # plugin's privacy impact, and then uncomment this line. For more information,
  # see https://developer.apple.com/documentation/bundleresources/privacy_manifest_files
  # s.resource_bundles = {'aura_native_privacy' => ['aura_native/Sources/aura_native/PrivacyInfo.xcprivacy']}
end
