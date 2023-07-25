#
# Be sure to run `pod lib lint WhisperDiarization.podspec' to ensure this is a
# valid spec before submitting.
#
# Any lines starting with a # are optional, but their use is encouraged
# To learn more about a Podspec see https://guides.cocoapods.org/syntax/podspec.html
#

Pod::Spec.new do |s|
  s.name             = 'WhisperDiarization'
  s.version          = '0.4.7'
  s.summary          = 'A short description of WhisperDiarization.'

# This description is used to generate tags and improve search results.
#   * Think: What does it do? Why did you write it? What is the focus?
#   * Try to keep it short, snappy and to the point.
#   * Write the description between the DESC delimiters below.
#   * Finally, don't worry about the indent, CocoaPods strips it!

  s.description      = <<-DESC
TODO: Add long description of the pod here.
                       DESC

  s.homepage         = 'https://github.com/NeoWorldTeam/WhisperDiarization'
  # s.screenshots     = 'www.example.com/screenshots_1', 'www.example.com/screenshots_2'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'fuhao' => 'fangshiyu2@gmail.com' }
  s.source           = { :git => 'git@github.com:NeoWorldTeam/WhisperDiarization.git', :tag => s.version.to_s }
  # s.social_media_url = 'https://twitter.com/<TWITTER_USERNAME>'

  s.ios.deployment_target = '13.0'

  s.source_files = 'WhisperDiarization/Classes/**/*.{c,swift,m,mm,h}'
  
  
#  s.public_header_files = 'WhisperDiarization/Classes/algorithm/clustering/AggClusteringWrapper.h'

  

  
  s.vendored_frameworks = 'WhisperDiarization/Frameworks/**/*.framework'
  s.static_framework = true

  s.resource_bundles = {
    'WhisperDiarization' => ['WhisperDiarization/Assets/**/*']
  }
  
  
  
  s.pod_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64' }
  s.user_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64' }


  
  s.dependency 'SpeakerEmbeddingForiOS', '~>0.3.2'
  s.dependency 'ObjectMapper'
  s.dependency 'RosaKit'
  
end
